# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 17:51:52 2022

@author: kient
"""

LOC = ['0','1']
VIR = ['1','2','3']
STAGE = ['a','b','c','d']

NUMLOC = len(LOC)
NUMSEROTYPES = len(VIR)
NUMR = len(STAGE)

STARTI = NUMLOC * NUMSEROTYPES * NUMR
STARTJ = STARTI + NUMLOC * NUMSEROTYPES
STARTS = STARTJ + NUMLOC * NUMSEROTYPES
DIM = STARTS + NUMLOC;

print("STARTI = %d" % (STARTI))
print("STARTJ = %d" % (STARTJ))
print("STARTS = %d" % (STARTS))

for index in range(0,DIM):
    label = ''
    vir = (index / NUMR) % NUMSEROTYPES;
    stg = index % NUMR;
    if index < STARTI:
        loc = index / (NUMSEROTYPES * NUMR);
        label = str(int(loc)) + '-R'
    elif index < STARTJ:
        loc = ((index - STARTJ) / (NUMSEROTYPES)) % NUMLOC
        label = str(int(loc)) + '-I'
    elif index < STARTS:
        loc = ((index - STARTS) / (NUMSEROTYPES)) % NUMLOC
        label = str(int(loc)) + '-J'
    else:
        loc = index - STARTS
        label = str(int(loc)) + '-S'
    if index < STARTI:
        label += VIR[int(vir)] + STAGE[int(stg)]
    elif index < STARTS:
        vir = (index - NUMSEROTYPES*NUMR*NUMLOC) % NUMSEROTYPES
        label += VIR[int(vir)]

    print('f[%d] - %s' % (index, label))
    
index = 0
# for loc in range(0,NUMLOC):
#     for vir in range(0,NUMSEROTYPES):
#         for stg in range(0,NUMR):
#             index = loc * NUMSEROTYPES * NUMR + vir *NUMR + stg
#             print("loc %d vir %d stg %d index %d" % (loc, vir, stg, index))
#             for l in range(0,NUMLOC):
#                 for v in range(0,NUMSEROTYPES):
#                     print("sum_foi += ppc->sigma[%d][%d] "
#                           "* ppc->beta[%d] "
#                           "* stf "
#                           "* ppc->eta[%d][%d] "
#                           "* y[%d]"
#                           %
#                           (vir,v,v,loc,l,STARTI + NUMSEROTYPES*l + v));
#             print("\n")
            
            
# for loc in range(0,NUMLOC):
#     for vir in range(0,NUMSEROTYPES):
#         index = loc * NUMSEROTYPES + vir
#         print("loc %d vir %d index %d" % (loc, vir, index))
#         for l in range(0,NUMLOC):
#             print("foi_on_susc_single_virus += ppc->eta[%d][%d]"
#                     " * stf" 
#                     " * ppc->beta[%d]"
#                     " * y[%d]"
#                     %(loc,l,vir,STARTI + NUMSEROTYPES*l + vir));
#         print("\n")
 
for loc in range(0,NUMLOC):
    for vir in range(0,NUMSEROTYPES):
        index = loc * NUMSEROTYPES + vir
        print("loc %d vir %d" % (loc, vir))
        for l in range(0,NUMLOC):
            for v in range(0,NUMSEROTYPES):
                for s in range(0,NUMR):
                    print("inflow_from_recovereds += ppc->sigma[%d][%d] "
                          "* stf "
                          "* ppc->beta[%d] "
                          "* ppc->eta[%d][%d] "
                          "* y[%d] "
                          "* y[%d]"
                            %(vir,v,vir,loc,l,
                              STARTI + NUMSEROTYPES*l + vir,
                              NUMSEROTYPES*NUMR*loc + NUMR*v + s));
        print("\n")

# for loc in range(0,NUMLOC):
#     print("loc %d" % (loc))
#     for l in range(0,NUMLOC):
#         for v in range(0,NUMSEROTYPES):
#             print("foi_on_susc_all_viruses += ppc->eta[%d][%d] "
#                   "* stf "
#                   "* ppc->beta[%d]  "
#                   "* y[%d] "
#                   %(loc,l,v,
#                     STARTI + NUMSEROTYPES*l + v));
#     print("\n")
                    















