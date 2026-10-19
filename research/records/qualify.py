"""Audit explicit action/outcome records without silently dropping bad episodes."""
import argparse
from collections import Counter
from datetime import datetime
import json,math
from pathlib import Path


def timestamp(value):
    if not isinstance(value,str):raise ValueError('timestamp must be a string')
    t=datetime.fromisoformat(value.replace('Z','+00:00'))
    if t.tzinfo is None or t.utcoffset() is None:raise ValueError('timestamp needs timezone')
    return t


def qualify(records):
    ids=Counter(r.get('episode_id') for r in records if isinstance(r,dict) and isinstance(r.get('episode_id'),str))
    rows=[];coverage=Counter();changed=0;statuses=Counter()
    for index,r in enumerate(records):
        issues=[];warnings=[]
        def issue(code):
            if code not in issues:issues.append(code)
        def time(value,code):
            try:return timestamp(value)
            except (ValueError,TypeError):issue(code);return None
        if not isinstance(r,dict):
            rows.append(dict(index=index,episode_id=None,eligible_for_action_outcome_evaluation=False,issues=['record_not_object'],warnings=[]));continue
        eid=r.get('episode_id')
        for field in ['episode_id','plant_id','machine_id','context_label']:
            if not isinstance(r.get(field),str) or not r[field].strip():issue('missing_'+field)
        if isinstance(eid,str) and ids[eid]>1:issue('duplicate_episode_id')
        if r.get('schema_version')!=1:issue('unsupported_schema_version')
        decision=time(r.get('decision_at'),'invalid_decision_time')
        features=r.get('observations')
        if not isinstance(features,dict) or not features:issue('missing_observations')
        else:
            for name,f in features.items():
                if not isinstance(f,dict):issue('malformed_observation');continue
                available=time(f.get('available_at'),'invalid_observation_availability_time')
                if available is not None and decision is not None and available>decision:issue('observation_available_after_decision')
                value=f.get('value')
                if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):issue('invalid_observation_value')
        req=r.get('requested_action')
        req=req if isinstance(req,dict) else {}
        if not isinstance(req.get('action_id'),str) or not req['action_id'].strip():issue('missing_requested_action')
        requested=time(req.get('requested_at'),'invalid_request_time')
        if requested is not None and decision is not None and requested<decision:issue('request_precedes_decision')
        execution=r.get('execution');execution=execution if isinstance(execution,dict) else {}
        status=execution.get('status')
        statuses[str(status)]+=1
        if status!='executed':issue('execution_not_verified_as_executed')
        applied=None
        if status=='executed':
            if not isinstance(execution.get('action_id'),str) or not execution['action_id'].strip():issue('missing_applied_action')
            applied=time(execution.get('applied_at'),'invalid_applied_time')
            if applied is not None and requested is not None and applied<requested:issue('application_precedes_request')
            if req.get('action_id') and execution.get('action_id') and req['action_id']!=execution['action_id']:
                warnings.append('requested_and_applied_actions_differ');changed+=1
        outcome=r.get('outcome');outcome=outcome if isinstance(outcome,dict) else {}
        value=outcome.get('value')
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):issue('missing_or_invalid_outcome')
        if not isinstance(outcome.get('metric'),str) or not outcome.get('metric'):issue('missing_outcome_metric')
        if not isinstance(outcome.get('unit'),str) or not outcome.get('unit'):issue('missing_outcome_unit')
        observed=time(outcome.get('observed_at'),'invalid_outcome_time')
        if observed is not None and applied is not None and observed<applied:issue('outcome_precedes_application')
        if outcome.get('censored') is not False:issue('outcome_censoring_unresolved')
        provenance=r.get('provenance')
        if not isinstance(provenance,dict) or not provenance.get('source') or not provenance.get('label_method'):issue('missing_provenance')
        eligible=not issues
        if eligible:coverage[(r['machine_id'],r['context_label'],execution['action_id'],outcome['metric'],outcome['unit'])]+=1
        rows.append(dict(index=index,episode_id=eid,eligible_for_action_outcome_evaluation=eligible,issues=issues,warnings=warnings))
    return dict(scope='Record consistency only. Eligibility does not establish causal identification, calibrated confidence, or safe control. Rejected/unknown executions remain in the audit; excluding them from estimation can introduce selection bias.',total_records=len(records),eligible_records=sum(r['eligible_for_action_outcome_evaluation'] for r in rows),execution_status_counts=dict(statuses),requested_applied_mismatches=changed,issue_counts=dict(Counter(code for r in rows for code in r['issues'])),observed_cell_counts=[dict(machine_id=k[0],context_label=k[1],applied_action=k[2],metric=k[3],unit=k[4],n=v) for k,v in sorted(coverage.items())],records=rows)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('input',type=Path);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    records=[]
    for i,line in enumerate(args.input.read_text().splitlines(),1):
        if line.strip():
            try:records.append(json.loads(line))
            except json.JSONDecodeError as e:raise ValueError(f'Invalid JSON on line {i}') from e
    report=qualify(records);args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['records','observed_cell_counts']},indent=2))
if __name__=='__main__':main()
