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
    transfer_verified: bool = False
    evidence: dict = field(default_factory=dict)

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
    interaction_at: float | None = None
    ended: float | None = None
    attempts: list = field(default_factory=list)
    placement_method: str | None = None
    placement_evidence: dict = field(default_factory=dict)
    clip_ended: float | None = None
    packet_opening_observations: list = field(default_factory=list)
    packet_opening_at_placement: dict | None = None


class PackingFlow:
    def __init__(self, ids):
        self.jewels = {jid: Jewel(jid) for jid in ids}
        self.active_jewel_id = None
        self.packet_opening_model = {'status':'not_loaded'}
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
        self.sealing_started_at = None
        self.bed_verified_at = None
        self.strip_removed = False
        self.strip_removed_at = None
        self.strip_absent_hits = 0
        self.strip_absent_since = None
        self.strip_clear_seconds = 0.
        self.strip_last_clear = None
        self.strip_previous_clear = False
        self.sealing_diagnostic = {}
        self.sealing_blocker_counts = {}
        self.message = "Pick up one jewel; its recording starts automatically."
        self.last_t = -math.inf

    def _reset_strip_progress(self):
        self.strip_absent_hits, self.strip_absent_since = 0, None
        self.strip_clear_seconds = 0.
        self.strip_last_clear, self.strip_previous_clear = None, False

    @property
    def all_packed(self):
        return bool(self.jewels) and all(j.status == INSIDE for j in self.jewels.values())

    @property
    def unconfirmed_placements(self):
        return [j.id for j in self.jewels.values() if j.status != INSIDE]

    @property
    def placement_warning(self):
        ids = self.unconfirmed_placements
        return ' Placement unconfirmed: '+', '.join(f'J{jid}' for jid in ids)+'.' if ids else ''

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
    def _latch(j, t, obs):
        j.status, j.entry, j.ended, j.placement_method = INSIDE, t, t, obs.method
        j.placement_evidence = dict(obs.evidence)
        route = "Moving gold from this pickup" if obs.method == "pickup_owned_gold_packet_interaction" else "Continuously tracked pickup hand"
        j.note = f"{route} crossed the packet boundary. Placement inferred; physical mouth/hidden insertion not verified."

    def update(self, t, observations, *, packet_ok, hands_clear, strip,
               pickup=None, identities_ready=True, packet_stationary=False,
               closure="model_missing", closure_confidence=None,
               packet_context_ok=None, strip_view_ok=True, reference_lost=False):
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
        observation_gap = gap > 2.5 and math.isfinite(gap)
        if observation_gap:
            self._reset_strip_progress()
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

        self.strip_hits = self.strip_hits + 1 if context_ok and strip == "present" else 0
        if self.strip_hits >= 1:
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
                if obs.transfer_verified:
                    self._latch(j, t, obs)
                else:
                    j.status = AT_PACKET
                    if j.interaction_at is None:
                        j.interaction_at = t
                    j.note = "Packet interaction observed; insertion unconfirmed. Waiting for continuous arrival or pickup-owned moving gold."

        # One-way strip workflow: first validated strip detection -> removal.
        # Packet detector dropouts never send an armed stage back to its start.
        if not self.strip_removal_requested and context_ok and strip == "present":
            self.strip_removal_requested = True
            self.sealing_started_at = t
            self.bed_verified_at = t if packet_ok and packet_stationary else None
            self.recording_required = self.packet_started = True
        if self.strip_removal_requested and not self.strip_removed:
            blocker = ('packet_not_detected' if not context_ok else
                       'strip_area_not_visible' if not strip_view_ok else
                       'hand_covering_strip_area' if not hands_clear else
                       'strip_still_detected' if strip == 'present' else
                       'strip_prediction_uncertain' if strip != 'absent' else 'checking_removal')
            paused = False
            if blocker == 'checking_removal':
                if self.strip_last_clear is not None and t-self.strip_last_clear > .8 and not self.strip_previous_clear:
                    self._reset_strip_progress()
                if self.strip_previous_clear and self.strip_last_clear is not None:
                    self.strip_clear_seconds += t-self.strip_last_clear
                self.strip_absent_hits += 1
                self.strip_absent_since = t if self.strip_absent_since is None else self.strip_absent_since
                self.strip_last_clear, self.strip_previous_clear = t, True
                if self.strip_absent_hits >= 5 and self.strip_clear_seconds >= 1.:
                    self.strip_removed, self.strip_removed_at = True, t
                    blocker = 'removal_confirmed'
            elif (blocker == 'strip_prediction_uncertain' and self.strip_last_clear is not None
                  and t-self.strip_last_clear <= .8):
                # Short ambiguity pauses progress. It supplies no hits or clear time.
                paused = True
                self.strip_previous_clear = False
            else:
                self._reset_strip_progress()
            self.sealing_blocker_counts[blocker] = self.sealing_blocker_counts.get(blocker, 0)+1
            self.sealing_diagnostic = dict(timestamp=t, blocker=blocker, strip=strip,
                packet_context=bool(context_ok), strip_area_visible=bool(strip_view_ok),
                hands_clear=bool(hands_clear), absent_hits=self.strip_absent_hits,
                required_hits=5, clear_seconds=self.strip_clear_seconds, required_clear_seconds=1.,
                uncertainty_paused=paused, observation_gap_reset=observation_gap,
                observation_gap_seconds=gap if math.isfinite(gap) else None)
        # The agreed process infers sealing from the verified strip-removal
        # sequence. No separate closure classifier or additional wait is used.
        if self.strip_seen and self.strip_removal_requested and self.strip_removed:
            self.sealed = True
            self.sealing_method = "strip_removal_inferred"
            self.closure_status = "inferred_from_strip_removal"
        if self.sealed:
            self.message = "Sealed — inferred from strip removal. Result retained. Saving PDF and videos."+self.placement_warning
        elif self.strip_removal_requested:
            labels = dict(packet_not_detected='Packet not detected. Keep the packet inside the bench ROI.',
                strip_area_not_visible='Strip area not visible. Unfold or reposition the packet.',
                hand_covering_strip_area='Hand covering the strip area. Move both hands away.',
                strip_still_detected='Strip still detected on the packet. Remove it and move the loose strip away.',
                strip_prediction_uncertain='Strip prediction uncertain. Hold the packet still with even lighting.',
                checking_removal='Checking strip removal.')
            diagnostic = self.sealing_diagnostic
            self.message = (('Start sealing: remove the sealing strip. ' if self.sealing_started_at == t else 'Strip-removal stage retained. ')
                +labels.get(diagnostic.get('blocker'), 'Waiting for strip verification.')
                +f" Clear observations: {self.strip_absent_hits}/5; clear time: {self.strip_clear_seconds:.1f}/1.0 s."
                +(' Progress paused briefly.' if diagnostic.get('uncertainty_paused') else '')
                +self.placement_warning)
        elif self.all_packed:
            self.message = "Show the packet with its sealing strip visible to start sealing."
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
            self.message = f"J{self.active_jewel_id} taken; placement not confirmed. Transfer retained. Show the packet strip to start sealing."
        else:
            self.message = "Pick up one jewel and move it into the detected full-bag ROI. Recording is automatic."
