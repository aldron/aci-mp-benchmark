import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from bench import ACI,make_data

def test_build_search_and_budgets():
 x,q=make_data(300,4,42,'gaussian')
 for gs in [2,4,8]:
  ix=ACI(x,gs,8)
  ids,dist,stats=ix.search(q[0],10,100,30)
  assert len(ids)==10 and len(set(ids[ids>=0]))==len(ids[ids>=0])
  assert stats[0]<=100 and stats[2]<=30
  for id,d in zip(ids,dist):
   if id>=0:assert np.isclose(d,np.sum((x[id]-q[0])**2),rtol=1e-4)
  assert ix.memory()>0
  ix.close()

def test_exact_query_when_present():
 x,_=make_data(500,1,12,'normalized');ix=ACI(x,4,16)
 ids,ds,st=ix.search(x[17],10,20000,1000)
 assert 17 in ids and ds[np.where(ids==17)[0][0]]<1e-6
 ix.close()
