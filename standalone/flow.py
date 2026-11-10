"""Pickup-owned surveillance; placement is retained without reidentification."""
from dataclasses import dataclass, field
import math

INSIDE = "Placed in packet — inferred"
PICKED = "Taken / interacting"
INTERACTING = "Taken / interaction retained"
UNVERIFIED = "Taken / placement unverified"
AT_PACKET = "At packet"


@dataclass
class Observation:
    point: tuple[float, float]
    packet_overlap: float = 0.
    identity_ok: bool = True
    near_packet: bool = False
    stationary_tolerance: float = 10.
    method: str = "pickup_owned_packet_interaction"

    @property
    def outside_packet(self):
        return self.packet_overlap <= .1 and not self.near_packet


@dataclass
class Pickup:
    active: bool = False
    jewel_id: int | None = None
    ambiguous: bool = False
    held_ids: tuple[int, ...] = ()


@dataclass
class Jewel:
    id: int
    status: str = "On bench"
    note: str = "Waiting for automatic pickup detection."
    last_seen: float = 0.
    started: float | None = None
    entry: float | None = None
    ended: float | None = None
    attempts: list = field(default_factory=list)
    placement_method: str | None = None
    clip_ended: float | None = None


class PackingFlow:
    def __init__(self, ids):
        self.jewels = {jid: Jewel(jid) for jid in ids}
        self.active_jewel_id = None
        self.packet_started = False  # Freeze assignment/geometry from first pickup.
        self.packet_seen = False
        self.waiting_for_packet = False
        self.recording_required = False
        self.unassigned_pickups = []
        self._ambiguous_ids = None
        self.strip_seen = False
        self.strip_hits = 0
        self.sealed = False
        self.sealing_method = None
        self.packet_number = dict(status="not_started", text=None, confidence=None, confirmed_at=None)
        self.closure_status = "not_checked"
        self.closure_confidence = None
        self.packet_stationary = False
        self.strip_removal_requested = False
        self.bed_verified_at = None
        self.strip_removed = False
        self.strip_removed_at = None
        self.strip_absent_hits = 0
        self.strip_absent_since = None
        self.message = "Pick up one jewel; its recording starts automatically."
        self.last_t = -math.inf

    @property
    def all_packed(self):
        return bool(self.jewels) and all(j.status == INSIDE for j in self.jewels.values())

    def uncertain(self, j, reason):
        if j.status == INSIDE:
            return
        j.note = reason  # Never discard a pickup-owned identity on occlusion.
        if j.started is not None:
            j.status = UNVERIFIED
        self.sealed = False

    def invalidate_packet(self, reason):
        if self.sealed:
            return
        self.message = reason
        # Camera/packet gaps affect evidence, not the historical pickup result.

    def pause_packet(self):
        """A detector dropout/moved packet pauses checks without losing progress."""
        if self.sealed:
            return
        self.packet_seen = self.waiting_for_packet = True
        self.packet_stationary = False
        self.strip_hits = 0
        self.closure_status, self.closure_confidence = "not_checked", None
        self.message = "Packet moving or temporarily hidden. Pickup and transfer stages retained."

    @staticmethod
    def _latch(j, t, method):
        j.status, j.entry, j.ended, j.placement_method = INSIDE, t, t, method
        route = "Moving gold from this pickup" if method == "pickup_owned_gold_packet_interaction" else "Pickup-associated hand/transfer"
        j.note = f"{route} reached the packet. Placement retained; contents inferred, identity inside is not rechecked."

    def update(self, t, observations, *, packet_ok, hands_clear, strip,
               pickup=None, identities_ready=True, packet_stationary=False,
               closure="model_missing", closure_confidence=None,
               packet_context_ok=None, strip_view_ok=True):
        if self.sealed or t <= self.last_t:
            return
        gap = t - self.last_t
        self.last_t = t
        pickup = pickup or Pickup()
        context_ok = packet_ok if packet_context_ok is None else packet_context_ok
        self.packet_seen = self.packet_seen or packet_ok
        self.waiting_for_packet = self.packet_seen and not packet_ok
        self.packet_stationary = bool(packet_stationary and packet_ok)
        self.closure_status, self.closure_confidence = "pending_strip_removal", None
        if gap > 2.5 and math.isfinite(gap):
            self.strip_absent_hits, self.strip_absent_since = 0, None
        if pickup.active:
            self.recording_required = self.packet_started = True
            if pickup.ambiguous or pickup.jewel_id not in self.jewels:
                key = tuple(sorted(pickup.held_ids))
                if key != self._ambiguous_ids:
                    self.unassigned_pickups.append(dict(timestamp=t, jewel_ids=list(key)))
                self._ambiguous_ids = key
            else:
                self._ambiguous_ids = None
                j = self.jewels[pickup.jewel_id]
                if j.started is None:
                    previous = self.jewels.get(self.active_jewel_id)
                    if previous is not None:
                        previous.clip_ended = t
                        if previous.status != INSIDE:
                            previous.status = INTERACTING
                            previous.note = "Pickup and transfer stage retained. Recording handed to the next jewel; packet visibility does not reset this jewel."
                    self.active_jewel_id = j.id
                    j.started = t
                    j.status, j.note = PICKED, "Pickup detected by Jewel Tracker; identity retained throughout this transfer."
        else:
            self._ambiguous_ids = None

        self.strip_hits = self.strip_hits + 1 if packet_ok and strip == "present" else 0
        if self.strip_hits >= 3:
            self.strip_seen = True
        for jid, j in self.jewels.items():
            if j.status == INSIDE:
                continue
            obs = observations.get(jid)
            if obs is not None:
                j.last_seen = t
            # An observation belongs to its pickup-owned transfer, including
            # the outgoing transfer on the recording-handoff frame.
            if j.started is None:
                continue
            if obs is not None and (obs.packet_overlap >= .1 or obs.near_packet):
                self._latch(j, t, obs.method)

        # One-way strip workflow: test-bed baseline -> removal -> removed.
        # Packet detector dropouts never send an armed stage back to its start.
        if (not self.strip_removal_requested and self.all_packed and self.strip_seen
                and packet_ok and packet_stationary and strip == "present"):
            self.strip_removal_requested, self.bed_verified_at = True, t
        if self.strip_removal_requested and not self.strip_removed:
            if context_ok and strip_view_ok and hands_clear and strip == "absent":
                self.strip_absent_hits += 1
                self.strip_absent_since = t if self.strip_absent_since is None else self.strip_absent_since
                if self.strip_absent_hits >= 5 and t-self.strip_absent_since >= 1.:
                    self.strip_removed, self.strip_removed_at = True, t
            else:
                self.strip_absent_hits, self.strip_absent_since = 0, None
        # The agreed process infers sealing from the verified strip-removal
        # sequence. No separate closure classifier or additional wait is used.
        if self.all_packed and self.strip_seen and self.strip_removal_requested and self.strip_removed:
            self.sealed = True
            self.sealing_method = "strip_removal_inferred"
            self.closure_status = "inferred_from_strip_removal"
        if self.sealed:
            self.message = "Sealed — inferred from strip removal. Result retained. Saving PDF and videos."
        elif self.all_packed:
            if self.strip_removal_requested:
                self.message = ("Strip removal in progress. Keep the strip area visible; packet/body context is enough."
                                if context_ok and strip_view_ok else
                                "Strip-removal stage retained. Waiting for a view of the strip area; full-bag detection is not required.")
            else:
                self.message = "Place the packet on the test bed with the strip visible for the initial strip check."
        elif self.waiting_for_packet:
            self.message = "Packet moving or temporarily hidden. Taken jewels and transfer stages retained; continue packing."
        elif any(j.status == INTERACTING for j in self.jewels.values()):
            self.message = "Pickup-owned interactions retained while recording the next jewel. No jewel identity check inside the packet."
        elif any(j.status == UNVERIFIED for j in self.jewels.values()):
            self.message = "Pickup retained; a transfer has no observed packet interaction. Review its recording or reset to repeat."
        elif not packet_ok:
            self.message = "Pickup recording is active; looking for the full bag ROI." if self.recording_required else "Waiting for pickup and automatic full-bag detection."
        elif not identities_ready:
            self.message = "Recording pickup events; SLIC/ResNet identities are preparing."
        elif self.active_jewel_id is not None and self.jewels[self.active_jewel_id].status != INSIDE:
            self.message = f"J{self.active_jewel_id} taken; packet entry not captured yet. Current hand/gold transfer retained. Strip stage waits for this transfer."
        else:
            self.message = "Pick up one jewel and move it into the detected full-bag ROI. Recording is automatic."
