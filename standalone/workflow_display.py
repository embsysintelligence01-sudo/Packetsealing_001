"""Independent pickup/rubbing/return columns for the experiment UI."""


def sequence_rows(rows,jewels,gold_masks,format_time):
    workflows={f'J{j.id}':getattr(j,'workflow',None) for j in jewels}
    workflows.update({item['track']:item.get('workflow') for item in gold_masks})
    result=[]
    for row in rows:
        identity,phase,cue,when,observed,verified,reason=row
        workflow=workflows.get(identity)
        if workflow:
            pickup=workflow.get('pickup_status','PENDING')
            rubbing=workflow.get('rubbing_status','PENDING')
            returned=workflow.get('return_status','PENDING')
            if workflow.get('identity_verified') is False:
                pickup='UNASSIGNED'
                observed=verified='unverified'
            else:
                observed=format_time(workflow.get('pickup_observed_at'))
                verified=format_time(workflow.get('pickup_verified_at'))
            number=workflow.get('event_number',1)
            phase=f'E{number} {phase}'
            previous=workflow.get('previous_event')
            if previous:
                reason=f"Previous E{previous['event_number']}: pickup {previous['pickup_status']}, rubbing {previous['rubbing_status']}. "+reason
        else:
            pickup,rubbing,returned=phase,'—','—'
        result.append([identity,pickup,rubbing,returned,cue,when,observed,verified,phase,reason])
    return result
