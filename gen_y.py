# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 17:51:52 2022

@author: kient
"""

LOC = ['0','1','2','3','4']
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
index_total = 0

# for loc in range(0,NUMLOC):
#     for vir in range(0,NUMSEROTYPES):
#         for stg in range(0,NUMR):
#             index = loc * NUMSEROTYPES * NUMR + vir *NUMR + stg
#             print("loc %d vir %d stg %d index %d" % (loc, vir, stg, index))
#             for l in range(0,NUMLOC):
#                 for v in range(0,NUMSEROTYPES):
#                     index_total = loc*NUMSEROTYPES*NUMR*NUMLOC*NUMSEROTYPES \
#                     + vir*NUMR*NUMLOC*NUMSEROTYPES + stg*NUMLOC*NUMSEROTYPES \
#                     + l*NUMSEROTYPES + v
#                     print("sum_foi[%d] += sigma[%d][%d] "
#                           "* beta[%d] "
#                           "* stf "
#                           "* eta[%d][%d] "
#                           "* y[%d]"
#                           %
#                           (index_total,vir,v,v,loc,l,STARTI + NUMSEROTYPES*l + v));
#             print("\n")
            
            
# for loc in range(0,NUMLOC):
#     for vir in range(0,NUMSEROTYPES):
#         index = STARTI + loc * NUMSEROTYPES + vir
#         print("loc %d vir %d index %d" % (loc, vir, index))
#         for l in range(0,NUMLOC):
#             print("foi_on_susc_single_virus += eta[%d][%d]"
#                     " * stf" 
#                     " * beta[%d]"
#                     " * y[%d]"
#                     %(loc,l,vir,STARTI + NUMSEROTYPES*l + vir));
#         print("f[%d] += y[%d] "
#               "* foi_on_susc_single_virus"
#               %
#               (STARTI + NUMSEROTYPES*loc + vir,STARTS + loc))
#         print("\n")
 
# for loc in range(0,NUMLOC):
#     for vir in range(0,NUMSEROTYPES):
#         index = loc * NUMSEROTYPES + vir
#         print("loc %d vir %d index %d" % (loc, vir, STARTI + index))
#         for l in range(0,NUMLOC):
#             for v in range(0,NUMSEROTYPES):
#                 for s in range(0,NUMR):
#                     index_total = loc*NUMSEROTYPES*NUMLOC*NUMSEROTYPES*NUMR \
#                                     + vir*NUMLOC*NUMSEROTYPES*NUMR \
#                                     + l*NUMSEROTYPES*NUMR + v*NUMR + s
#                     print("[%d] inflow_from_recovereds += sigma[%d][%d] "
#                           "* stf "
#                           "* beta[%d] "
#                           "* eta[%d][%d] "
#                           "* y[%d] "
#                           "* y[%d]"
#                             %(index_total,vir,v,vir,loc,l,
#                               STARTI + NUMSEROTYPES*l + vir,
#                               NUMSEROTYPES*NUMR*loc + NUMR*v + s));
#         print("\n")

for loc in range(0,NUMLOC):
    index = STARTS + loc
    print("loc %d index %d" % (loc,index))
    for l in range(0,NUMLOC):
        for v in range(0,NUMSEROTYPES):
            index_total = loc*NUMLOC*NUMSEROTYPES + l*NUMSEROTYPES + v;
            print("foi_on_susc_all_viruses[%d] += eta[%d][%d] "
                  "* stf "
                  "* beta[%d]  "
                  "* y[%d] "
                  %(index_total,loc,l,v,
                    STARTI + NUMSEROTYPES*l + v));
    print("\n")
   
# for loc in range(0,NUMLOC):
#     print("loc %d" % (loc))                 
#     for vir in range(0,NUMSEROTYPES):
#         print("f[%d] += trr "
#               "* y[%d]"
#               %
#               (STARTS + loc,NUMSEROTYPES*NUMR*loc + NUMR*vir + (NUMR-1)))














