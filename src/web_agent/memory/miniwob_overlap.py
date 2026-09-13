"""Train/evaluation overlap audit for the revised MiniWoB task binding."""
from dataclasses import dataclass
import re
from web_agent.runtime.contracts import canonical_sha256
from web_agent.runtime.duplicate_audit import DuplicateAuditError

def tokens(text):
    return tuple(re.findall(r'[a-z0-9]+',text.lower()))
def trigrams(words):
    return set(zip(words,words[1:],words[2:]))

class MiniWoBOverlapAudit:
    manifest_id='miniwob-train-goal-overlap-v1'
    def __init__(self,task,training_items):
        goal=tokens(task.goal)
        grams=trigrams(goal)
        matches=[]
        for item in training_items:
            source=item['material'];other=tokens(source['task_description']);othergrams=trigrams(other)
            similarity=len(grams&othergrams)/max(len(grams|othergrams),1)
            if source['canonical_task_id']==task.task_id or other==goal or (grams and othergrams and similarity>=.8):
                matches.append(source['canonical_task_id'])
        self.task_id=task.task_id
        self.record={'schema_version':'miniwob.train-overlap.v1','task_id':task.task_id,'goal_sha256':canonical_sha256(goal),'rule':'exact normalized goal or word-trigram Jaccard >=0.8 (conservative cross-site comparison)','training_item_count':len(training_items),'matching_source_tasks':sorted(set(matches))}
        self.manifest_sha256=canonical_sha256(self.record)
    def clusters_for(self,task_id,**kwargs):
        if task_id!=self.task_id:raise DuplicateAuditError('overlap audit belongs to another task')
        if self.record['matching_source_tasks']:raise DuplicateAuditError('evaluation task overlaps training memory')
        return ('miniwob-eval-'+self.record['goal_sha256'],)
    def entry_binding_sha256(self,task_id):
        self.clusters_for(task_id)
        return self.manifest_sha256
