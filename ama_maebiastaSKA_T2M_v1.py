# -*- coding: utf-8 -*-
"""
Created on Thu Aug 08 21:09:27 2013

@author: ama
"""

import sys
import numpy as np
import os
from os.path import exists
from sys import argv
import calendar
import datetime
import shutil

import matplotlib.pyplot as plt

NFLen = 49
fstatname  = np.chararray(NFLen,72) #67   # ---> statistics for all stations by forecast length
fcorfname  = np.chararray(NFLen,66) #61 45)    # ---> T2m correction values for all stations

#    fstat.write ('UTC FL   N  METS  #D %GOOD %OTR  %OK %BAD %TOT   %B+  %B- %HRo %HRn     avgC    maxC    minC    avgBo   avgBn   avgMo   avgMn   diffM    maxBo   maxBn   minBo   minBn   rngBo   rngBn \n') 
#        fstat.write ( ('%3s%3d%4d%6s%4d %5d%5d%5d%5d%5d %5d%5d%5d%5d %8.3f%8.3f%8.3f %8.3f%8.3f%8.3f%8.3f%8.3f %8.3f%8.3f%8.3f%8.3f%8.3f%8.3f \n') % \


######shutil.copy2(fcorfname[nn], '/new/dir/newname.ext')



# -----> SELECT NECESSARY PARAMETERS ------------------------------------------
cMODEL = 'SKA'   # SKA, GLB, DKA, K05

runYY = '15'    # 13 14 15 ... --- year
runMM = '03'    #  01 ... 12 --- month
runDD = '01'    # --- day
runHH = '00'    # --- UTC hour (00, 06, 12, 18 UTCs) -- K05 & SKA
                # --- UTC hour (03, 09, 15, 21 UTCs) -- GLB & DKA
                
startDAY = 1   # 1  ---> start day for a selected month
endDAY   = 31   # 31/32 or 29/30 (Feb)   ---> end day for a selected month

metstIDn = 6183 # 6183, 6072, 6017 
        
iflagDM = 0 # -- 0 - include results for all stations
            # -- 1 - consider decision-making tree (include results for selected stations)                    
            # -- 2 - consider only 1 selected station            
if iflagDM == 0: ainfo = 'ALL stations';             cinfo ='ALL'
if iflagDM == 1: ainfo = 'SELECTED (DM) stations';   cinfo ='DMS'
if iflagDM == 2: ainfo = 'Station ' + str(metstIDn); cinfo = str(metstIDn)

iflagMETPAR = 1 # --- 1 = T2m; 
                # --- 2 = W10m
if iflagMETPAR == 1: cMETPAR  = 'T2m'; cMPUNITS = '[deg C]'
if iflagMETPAR == 2: cMETPAR = 'W10m'; cMPUNITS = '[m/s]'

###############################################################################

# --- text for figures        
textUTChh = runHH # --- HIRLAM-SKA model runs at UTC times (00, 06, 12, 18 UTCs)
textYY = '20'+runYY
textMM = runMM

daydirM1 = '2015010100'; daydirM0 = '2015010100'
daydirM1s = np.chararray(100,10)
daydirM0s = np.chararray(100,10)

# -----------------------------------------------------------------------------
# -----> create dates for corrent & previous day runs
for kmk in range (startDAY, endDAY):    
    iiday = kmk
    FLm0d = iiday * 24 * 60 * 60 # --- minus N-day -- convert forecast length into seconds   
    vd1 = datetime.datetime(int('20'+runYY), int(runMM), int(runDD), int(runHH))
    vd2 = calendar.timegm(vd1.timetuple())
    vd3 = vd2 + FLm0d  # i.e. current date -0 day
    vd4 = datetime.datetime.utcfromtimestamp(vd3)
    val = str(vd4)
    daydirM0s[kmk] = '20' + str(val[2:4]) + str(val[5:7]) + str(val[8:10]) + str(val[11:13])

    vd3 = vd2 +  (iiday-1) * 24 * 60 * 60  # i.e. previous date -1 day
    vd4 = datetime.datetime.utcfromtimestamp(vd3)
    val = str(vd4)
    daydirM1s[kmk] = '20' + str(val[2:4]) + str(val[5:7]) + str(val[8:10]) + str(val[11:13])    
    
    print 'kmk, iiday, daydirM0s/M1s[kmk] =', kmk, iiday, daydirM0s[kmk], daydirM1s[kmk]
            

#sys.exit()

# -----------------------------------------------------------------------------
# -----> files to save extracted data for plotting by days for FLen -----------
line0 =' DAY + FL:     00     01     02     03     04     05     06     07     08     09     10'
line1 ='     11     12     13     14     15     16     17     18     19     20'
line2 ='     21     22     23     24     25     26     27     28     29     30'
line3 ='     31     32     33     34     35     36     37     38     39     40'
line4 ='     41     42     43     44     45     46     47     48\n'
lineall = line0 + line1 + line2 + line3 + line4

fmaefile = 'stats_'+cMODEL+'_'+textYY+textMM+textUTChh+'utc_mae_'+cMETPAR+'_matrix.dat'
fmaestat = open(fmaefile,'w')
fmaestat.write (lineall)

fhitrfile = 'stats_'+cMODEL+'_'+textYY+textMM+textUTChh+'utc_hitrate_'+cMETPAR+'_matrix.dat'
fhitrstat = open(fhitrfile,'w')
fhitrstat.write (lineall)

favgTcfile = 'stats_'+cMODEL+'_'+textYY+textMM+textUTChh+'utc_avg_'+cMETPAR+'_matrix.dat'
favgTcstat = open(favgTcfile,'w')
favgTcstat.write (lineall)

fminTcfile = 'stats_'+cMODEL+'_'+textYY+textMM+textUTChh+'utc_min_'+cMETPAR+'_matrix.dat'
fminTcstat = open(fminTcfile,'w')
fminTcstat.write (lineall)

fmaxTcfile = 'stats_'+cMODEL+'_'+textYY+textMM+textUTChh+'utc_max_'+cMETPAR+'_matrix.dat'
fmaxTcstat = open(fmaxTcfile,'w')
fmaxTcstat.write (lineall)

fbiasofile = 'stats_'+cMODEL+'_'+textYY+textMM+textUTChh+'utc_biaso_'+cMETPAR+'_matrix.dat'
fbiasostat = open(fbiasofile,'w')
fbiasostat.write (lineall)

fbiasnfile = 'stats_'+cMODEL+'_'+textYY+textMM+textUTChh+'utc_biasn_'+cMETPAR+'_matrix.dat'
fbiasnstat = open(fbiasnfile,'w')
fbiasnstat.write (lineall)

fokfile = 'stats_'+cMODEL+'_'+textYY+textMM+textUTChh+'utc_ok_'+cMETPAR+'_matrix.dat'
fokstat = open(fokfile,'w')
fokstat.write (lineall)

# --- to count summary stats over improvements
gAnsignMAE  = 0; bAnsignMAE  = 0 # -- for MAEs
gAnsignBIAS = 0; bAnsignBIAS = 0 # -- for BIASes
gAnsignHR   = 0; bAnsignHR   = 0 # -- for Hit-Rates



# -----------------------------------------------------------------------------
for kmk in range (startDAY, endDAY):    # --- START LOOP OVER DAYS IN MONTH
    
    # ----- define arrays for saving output in files
    difMAEfl = np.zeros(NFLen) # -- for MAEs by FLen
    difHRfl  = np.zeros(NFLen) # -- for Hit_rates by FLen
    AnavgTc  = np.zeros(NFLen) # -- for avgTc by FLen
    AnmaxTc  = np.zeros(NFLen) # -- for maxTc by FLen
    AnminTc  = np.zeros(NFLen) # -- for minTc by FLen 
    BIASnfl  = np.zeros(NFLen) # -- for new BIAS
    BIASofl  = np.zeros(NFLen) # -- for old BIAS
    AnPRokNB = np.zeros(NFLen) # -- for percentage of OK-corrected cases (bias improved)

    Nncm   = np.zeros(NFLen)  # -- count for maes cases
    Nncb   = np.zeros(NFLen)  # -- count for biases cases
    Nnhr   = np.zeros(NFLen)  # -- count for hit-rate cases
    Nnokc  = np.zeros(NFLen)  # -- count for OK-corrected cases
    Nntc   = np.zeros(NFLen)  # -- count for T corrs cases
        
    print 'date in a sequence: kmk =', kmk
    
    # -------------------------------------------------------------------------
    for nn in range (0,NFLen):  # --- START LOOP OVER FORECAST LENGTHS (00-48)
        nnlen = len(str(int(nn)))
        if nnlen == 1: adxNN = '0' + str(int(nn))               
        else: adxNN = str(int(nn))
        
        nnum2 = int(runHH) + nn
        nnlen2 = len(str(nnum2))
        if nnlen2 == 1: adxNN2 = '0' + str(int(nnum2))               
        else: adxNN2 = str(int(nnum2))        
        
# -E-> 49-NFLen files with saved avg/min/max/range statistics (on bias, mae, hit-rate, corr): ALL stations by ONE forecast length 
# ... DSTATSUMV/UTC00/2013053100/FH00UTC_2013053100_stats_fl00_00_00.out
        fstatname[nn]='data/DSTATSUMV/UTC'+runHH+'/' + daydirM0s[kmk] + '/FH'+runHH+'UTC_'\
        + daydirM0s[kmk] + '_stats_fl' + adxNN + '_'+runHH+'_' + adxNN2 + '.out'
        #t print 'nn, fstatname[nn] =', nn, fstatname[nn]

# -F-> 49-NFLen files to save T2m correction values: ALL stations by ONE forecast length           
        fcorfname[nn] = 'data/DINTERPOL/UTC'+runHH+'/' + daydirM1s[kmk] + 'R/FH'+runHH+'UTC_'\
        + 'corta_fl' + adxNN + '_'+runHH+'_' + adxNN2 + '.out'
        #t print 'nn, fcorfname[nn] =', nn, fcorfname[nn]
        
# ---> Reading statistics on by forecast lengths
        # ('UTC FL   N  METS  #D %GOOD %OTR  %OK %BAD %TOT   %B+  %B- %HRo %HRn     avgC    maxC    minC    avgBo   avgBn   avgMo   avgMn   diffM    maxBo   maxBn   minBo   minBn   rngBo   rngBn \n') 
        dataFL = np.genfromtxt(fstatname[nn], dtype = float, skip_header=2, skip_footer=2)
        #t print 'dataFL[:] =', dataFL[:]
        nptsFL = 0; nptsFL = dataFL.shape[0]
        #t print '     nn, fstatname[nn], nptsFL =', nn, fstatname[nn], nptsFL
        
        # -- assign loaded data with corresponding parameters and convert from-list-to-array        
        FHUTC        = np.array(dataFL [0:nptsFL,  0])  # UTC term
        sRRstatid    = np.array(dataFL [0:nptsFL,  3])  # station ID
        stavgi       = np.array(dataFL [0:nptsFL,  4])  # number days used in statistics 
        stPRgoodNB   = np.array(dataFL [0:nptsFL,  5])  # %GOOD 
        stPRotherNB  = np.array(dataFL [0:nptsFL,  6])  # %OTHER
        stPRokNB     = np.array(dataFL [0:nptsFL,  7])  # %OK
        stPRbadNB    = np.array(dataFL [0:nptsFL,  8])  # %BAD 
        stPRtotNB    = np.array(dataFL [0:nptsFL,  9])  # %TOT    
        stPRnegNB    = np.array(dataFL [0:nptsFL, 10])  # %B+  
        stPRposNB    = np.array(dataFL [0:nptsFL, 11])  # %B- 
        stPRnHRo     = np.array(dataFL [0:nptsFL, 12])  # %HRo 
        stPRnHRn     = np.array(dataFL [0:nptsFL, 13])  # %HRn
        stavgTc      = np.array(dataFL [0:nptsFL, 14])  # avgC       
        stmaxTc      = np.array(dataFL [0:nptsFL, 15])  # maxC   
        stminTc      = np.array(dataFL [0:nptsFL, 16])  # minC 
        stavgBo      = np.array(dataFL [0:nptsFL, 17])  # avgBo     
        stavgBn      = np.array(dataFL [0:nptsFL, 18])  # avgBn   
        stavgMo      = np.array(dataFL [0:nptsFL, 19])  # avgMo 
        stavgMn      = np.array(dataFL [0:nptsFL, 20])  # avgMn   
        stdifMon     = np.array(dataFL [0:nptsFL, 21])  # diffM    
        stmaxBo      = np.array(dataFL [0:nptsFL, 22])  # maxBo   
        stmaxBn      = np.array(dataFL [0:nptsFL, 23])  # maxBn   
        stminBo      = np.array(dataFL [0:nptsFL, 24])  # minBo   
        stminBn      = np.array(dataFL [0:nptsFL, 25])  # minBn   
        strngBo      = np.array(dataFL [0:nptsFL, 26])  # rngBo   
        strngBn      = np.array(dataFL [0:nptsFL, 27])  # rngBn
        
        #t print 'sRRstatid =', sRRstatid
        misval = -99.99
    
        valmask0 = stdifMon == misval 
        #t print 'valmask0 =', valmask0
        maskstdifMon = np.ma.masked_array(stdifMon, mask=valmask0)
        #t print 'maskstdifMon =', maskstdifMon
        stavgi = np.ma.count(maskstdifMon)             # -- count N-non-missing values
        #t print ' stavgi =', stavgi
        AVstdifMon = np.ma.mean(maskstdifMon)
        MIstdifMon = np.ma.min(maskstdifMon)
        MAstdifMon = np.ma.max(maskstdifMon)

        maskstavgMo = np.ma.masked_array(stavgMo, mask=valmask0)
        AVstavgMo = np.ma.mean(maskstavgMo)
        maskstavgMn = np.ma.masked_array(stavgMn, mask=valmask0)
        AVstavgMn = np.ma.mean(maskstavgMn)   
    
        #t print ' stdifMon: AVG, MIN, MAX =', (('%3d%8.3f%8.3f%8.3f%4d%8.3f%8.3f') % (nn, AVstdifMon,MIstdifMon,MAstdifMon, stavgi, AVstavgMo, AVstavgMn))
    
        AVstdifMon = np.mean(stdifMon)
        # print ' AAstdifMon =', AVstdifMon

# -----------------------------------------------------------------------------
# ---> Reading corrections for T2m by forecast lengths
        ## ('D      Ta-cor      Ta-cor  Latitude Longitude D     stID    Height    LSmask \n')
        dataCF = np.genfromtxt(fcorfname[nn], dtype = float) #, skip_header=2, skip_footer=2)
        #t print 'dataCF[:] =', dataCF[:]
        nptsCF = 0; nptsCF = dataCF.shape[0]
        #t print '     nn, fstatname[nn], nptsCF =', nn, fstatname[nn], nptsCF
    
        CFscort2m = np.zeros(nptsCF)
        CFmetstid = np.zeros(nptsCF)
        # -- assign loaded data with corresponding parameters and convert from-list-to-array        
        CFscort2m        = np.array(dataCF [0:nptsCF,  2])  # temperature correction
        CFmetstid        = np.array(dataCF [0:nptsCF,  6])  # station ID
   
        #t if nn == 48: print 'CFscort2m =', CFscort2m
        #t print 'CFmetstid =', CFmetstid

        #fcorf.write (('%1s%12.3f%12.3f%9s%1s%9s%1s%2s%9s%5s%5s%10s\n') % \
        #(dumv1, scort2m, scort2m, stlat, extra0, stlon, extra0, dumv2, metstid, height, extra00, landmask))


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++    
# ==1a== MAE EVALUATION =======================================================        
        ncm = 0
        sumMAEo = 0.
        sumMAEn = 0.        
        
        for jFL in range (0,nptsFL):       
            for jCF in range (0,nptsCF):
                MAEsign = 0
                # --- consider decision-making tree excluding some stations
                if iflagDM == 1 and sRRstatid[jFL] == CFmetstid[jCF] and CFscort2m[jCF] != 0.:  
                    if stavgMn[jFL] < stavgMo[jFL]: MAEsign = 1
                    ncm = ncm + 1
                    sumMAEo = sumMAEo + stavgMo[jFL]
                    sumMAEn = sumMAEn + stavgMn[jFL]
                # --- including all stations
                elif iflagDM == 0 and sRRstatid[jFL] == CFmetstid[jCF]:
                    if stavgMn[jFL] < stavgMo[jFL]: MAEsign = 1
                    ncm = ncm + 1
                    sumMAEo = sumMAEo + stavgMo[jFL]
                    sumMAEn = sumMAEn + stavgMn[jFL]
                 # --- including only one selected station
                elif iflagDM == 2 and sRRstatid[jFL] == metstIDn:
                    if stavgMn[jFL] < stavgMo[jFL]: MAEsign = 1
                    ncm = ncm + 1
                    sumMAEo = sumMAEo + stavgMo[jFL]
                    sumMAEn = sumMAEn + stavgMn[jFL]                   
                else:
                    pass
                    
        if ncm == 0:    # -- if number of cases = 0        
            AnMAEo = 0.
            print '??? number of cases = 0'
        else: 
            AnMAEo = sumMAEo/ncm
        if ncm == 0: 
            AnMAEn = 0.
            print '??? number of cases = 0'
        else: 
            AnMAEn = sumMAEn/ncm
        
        AnsignMAE = 0
        difMAE  = AnMAEo - AnMAEn   # --- find MAE difference OLD-NEW
        difMAEfl[nn] = difMAE
        Nncm[nn]     = ncm
        
        if AnMAEn <= AnMAEo:    # --- new MAE is better than old MAE
            AnsignMAE = 1
            gAnsignMAE = gAnsignMAE + 1 # --- improved
        else:
            bAnsignMAE = bAnsignMAE + 1 # --- not improved
            
        #t print 'daydirM1s, nn, ncm, AnMAEo, AnMAEn, difMAE, AnsignMAE =',\
        #t (('%10d%3d%4d%8.3f%8.3f%8.2f%2d') % (int(daydirM1s[kmk]), nn, ncm, AnMAEo, AnMAEn, difMAE, AnsignMAE))

        
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++    
# ==1b== BIAS EVALUATION ========================================================        
        ncb = 0
        sumBIASo = 0.
        sumBIASn = 0.
                
        for jFL in range (0,nptsFL):       
            for jCF in range (0,nptsCF):
                BIASsign = 0
                # --- consider decision-making tree excluding some stations
                if iflagDM == 1 and sRRstatid[jFL] == CFmetstid[jCF] and CFscort2m[jCF] != 0.:
                    if (abs(stavgBn[jFL])) < (abs(stavgBo[jFL])): BIASsign = 1      
                    ncb = ncb + 1
                    sumBIASo = sumBIASo + stavgBo[jFL]
                    sumBIASn = sumBIASn + stavgBn[jFL]
                # --- including all
                elif iflagDM == 0 and sRRstatid[jFL] == CFmetstid[jCF]:
                    if (abs(stavgBn[jFL])) < (abs(stavgBo[jFL])): BIASsign = 1      
                    ncb = ncb + 1
                    sumBIASo = sumBIASo + stavgBo[jFL]
                    sumBIASn = sumBIASn + stavgBn[jFL]
                 # --- including only one selected station
                elif iflagDM == 2 and sRRstatid[jFL] == metstIDn:
                    if (abs(stavgBn[jFL])) < (abs(stavgBo[jFL])): BIASsign = 1      
                    ncb = ncb + 1
                    sumBIASo = sumBIASo + stavgBo[jFL]
                    sumBIASn = sumBIASn + stavgBn[jFL]                    
                else:
                    pass
                    
        if ncb == 0: 
            AnBIASo = 0.
        else: 
            AnBIASo = sumBIASo/ncb
        if ncb == 0: 
            AnBIASn = 0.
        else: 
            AnBIASn = sumBIASn/ncb
        
        BIASnfl[nn] = AnBIASn
        BIASofl[nn] = AnBIASo
        Nncb[nn]    = ncb

        if (abs(AnBIASn)) <= (abs(AnBIASo)):
            AnsignBIAS = 1
            gAnsignBIAS = gAnsignBIAS + 1
        else:
            bAnsignBIAS = bAnsignBIAS + 1
            
        
# ==2== HIT-RATE EVALUATION ===================================================
        nhr = 0
        sumHRo = 0.
        sumHRn = 0.
        
        for jFL in range (0,nptsFL):
            for jCF in range (0,nptsCF):
                HRsign = 0
                # --- consider decision-making tree excluding some stations
                if iflagDM == 1 and sRRstatid[jFL] == CFmetstid[jCF] and CFscort2m[jCF] != 0.:
                    if stPRnHRo[jFL] < stPRnHRn[jFL]: HRsign = 1
                    nhr = nhr + 1
                    sumHRo = sumHRo + stPRnHRo[jFL]
                    sumHRn = sumHRn + stPRnHRn[jFL]
                # --- including all stations
                elif iflagDM == 0 and sRRstatid[jFL] == CFmetstid[jCF]:
                    if stPRnHRo[jFL] < stPRnHRn[jFL]: HRsign = 1
                    nhr = nhr + 1
                    sumHRo = sumHRo + stPRnHRo[jFL]
                    sumHRn = sumHRn + stPRnHRn[jFL]
                 # --- including only one selected station
                elif iflagDM == 2 and sRRstatid[jFL] == metstIDn:
                    if stPRnHRo[jFL] < stPRnHRn[jFL]: HRsign = 1
                    nhr = nhr + 1
                    sumHRo = sumHRo + stPRnHRo[jFL]
                    sumHRn = sumHRn + stPRnHRn[jFL]                    
                else:
                    pass

        if nhr == 0:
            AnHRo = 0.
        else: 
            AnHRo = sumHRo/nhr
        if nhr == 0: 
            AnHRn = 0.
        else: 
            AnHRn = sumHRn/nhr
            
        AnsignHR = 0
        difHR  = AnHRn - AnHRo
        difHRfl[nn] = difHR
        Nnhr[nn]    = nhr
            
        if AnHRn >= AnHRo: 
            AnsignHR = 1
            gAnsignHR = gAnsignHR + 1
        else:
            bAnsignHR = bAnsignHR + 1
                
                
# ==3=== AVG/MAX/MIN CORRECTION EVALUATION ====================================
        ntc = 0
        sumavgTc = 0.
        summaxTc = 0.
        summinTc = 0.
        
        for jFL in range (0,nptsFL):
            for jCF in range (0,nptsCF):
                
                if sRRstatid[jFL] == CFmetstid[jCF]:    # --- CONSIDER ALL CASES
                    ntc = ntc + 1
                    sumavgTc = sumavgTc + stavgTc[jFL]
                    summaxTc = summaxTc + stmaxTc[jFL]
                    summinTc = summinTc + stminTc[jFL]
                    
        AnavgTc[nn] = sumavgTc/ntc
        AnmaxTc[nn] = summaxTc/ntc
        AnminTc[nn] = summinTc/ntc
        Nntc[nn]    = ntc


# ==3=== PERCENTAGE OF -OK- CORRECTED CASES FOR BIAS EVALUATION ===============
        nokc = 0
        sumPRokNB = 0.
        
        for jFL in range (0,nptsFL):
            for jCF in range (0,nptsCF):    
                
                if sRRstatid[jFL] == CFmetstid[jCF]:   # --- CONSIDER ALL CASES
                    nokc = nokc + 1
                    sumPRokNB = sumPRokNB + stPRokNB[jFL]
                    
        AnPRokNB[nn] = sumPRokNB/nokc
        Nnokc[nn] = nokc




# ----- WRITING OUTPUTS -------------------------------------------------------

    # --- write output for percentage of corrected cases
    fokstat.write( ('%10d%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f\n') \
    % (int(daydirM1s[kmk]),AnPRokNB[0], AnPRokNB[1],AnPRokNB[2],AnPRokNB[3],AnPRokNB[4],AnPRokNB[5],AnPRokNB[6],AnPRokNB[7],\
    AnPRokNB [8],AnPRokNB [9],AnPRokNB[10],AnPRokNB[11],AnPRokNB[12],AnPRokNB[13],AnPRokNB[14],AnPRokNB[15],AnPRokNB[16],AnPRokNB[17],\
    AnPRokNB[18],AnPRokNB[19],AnPRokNB[20],AnPRokNB[21],AnPRokNB[22],AnPRokNB[23],AnPRokNB[24],AnPRokNB[25],AnPRokNB[26],AnPRokNB[27],\
    AnPRokNB[28],AnPRokNB[29],AnPRokNB[30],AnPRokNB[31],AnPRokNB[32],AnPRokNB[33],AnPRokNB[34],AnPRokNB[35],AnPRokNB[36],AnPRokNB[37],\
    AnPRokNB[38],AnPRokNB[39],AnPRokNB[40],AnPRokNB[41],AnPRokNB[42],AnPRokNB[43],AnPRokNB[44],AnPRokNB[45],AnPRokNB[46],AnPRokNB[47],\
    AnPRokNB[48]) )
    
    # --- write output for old bias
    fbiasostat.write( ('%10d%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f\n') \
    % (int(daydirM1s[kmk]),BIASofl[0], BIASofl[1],BIASofl[2],BIASofl[3],BIASofl[4],BIASofl[5],BIASofl[6],BIASofl[7],\
    BIASofl [8],BIASofl [9],BIASofl[10],BIASofl[11],BIASofl[12],BIASofl[13],BIASofl[14],BIASofl[15],BIASofl[16],BIASofl[17],\
    BIASofl[18],BIASofl[19],BIASofl[20],BIASofl[21],BIASofl[22],BIASofl[23],BIASofl[24],BIASofl[25],BIASofl[26],BIASofl[27],\
    BIASofl[28],BIASofl[29],BIASofl[30],BIASofl[31],BIASofl[32],BIASofl[33],BIASofl[34],BIASofl[35],BIASofl[36],BIASofl[37],\
    BIASofl[38],BIASofl[39],BIASofl[40],BIASofl[41],BIASofl[42],BIASofl[43],BIASofl[44],BIASofl[45],BIASofl[46],BIASofl[47],\
    BIASofl[48]) )
    
    # --- write output for new bias
    fbiasnstat.write( ('%10d%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f\n') \
    % (int(daydirM1s[kmk]),BIASnfl[0], BIASnfl[1],BIASnfl[2],BIASnfl[3],BIASnfl[4],BIASnfl[5],BIASnfl[6],BIASnfl[7],\
    BIASnfl [8],BIASnfl [9],BIASnfl[10],BIASnfl[11],BIASnfl[12],BIASnfl[13],BIASnfl[14],BIASnfl[15],BIASnfl[16],BIASnfl[17],\
    BIASnfl[18],BIASnfl[19],BIASnfl[20],BIASnfl[21],BIASnfl[22],BIASnfl[23],BIASnfl[24],BIASnfl[25],BIASnfl[26],BIASnfl[27],\
    BIASnfl[28],BIASnfl[29],BIASnfl[30],BIASnfl[31],BIASnfl[32],BIASnfl[33],BIASnfl[34],BIASnfl[35],BIASnfl[36],BIASnfl[37],\
    BIASnfl[38],BIASnfl[39],BIASnfl[40],BIASnfl[41],BIASnfl[42],BIASnfl[43],BIASnfl[44],BIASnfl[45],BIASnfl[46],BIASnfl[47],\
    BIASnfl[48]) )
    
    # --- write output for minTc
    fminTcstat.write( ('%10d%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f\n') \
    % (int(daydirM1s[kmk]),AnminTc[0], AnminTc[1],AnminTc[2],AnminTc[3],AnminTc[4],AnminTc[5],AnminTc[6],AnminTc[7],\
    AnminTc [8],AnminTc [9],AnminTc[10],AnminTc[11],AnminTc[12],AnminTc[13],AnminTc[14],AnminTc[15],AnminTc[16],AnminTc[17],\
    AnminTc[18],AnminTc[19],AnminTc[20],AnminTc[21],AnminTc[22],AnminTc[23],AnminTc[24],AnminTc[25],AnminTc[26],AnminTc[27],\
    AnminTc[28],AnminTc[29],AnminTc[30],AnminTc[31],AnminTc[32],AnminTc[33],AnminTc[34],AnminTc[35],AnminTc[36],AnminTc[37],\
    AnminTc[38],AnminTc[39],AnminTc[40],AnminTc[41],AnminTc[42],AnminTc[43],AnminTc[44],AnminTc[45],AnminTc[46],AnminTc[47],\
    AnminTc[48]) )
    
    # --- write output for maxTc
    fmaxTcstat.write( ('%10d%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f\n') \
    % (int(daydirM1s[kmk]),AnmaxTc[0], AnmaxTc[1],AnmaxTc[2],AnmaxTc[3],AnmaxTc[4],AnmaxTc[5],AnmaxTc[6],AnmaxTc[7],\
    AnmaxTc [8],AnmaxTc [9],AnmaxTc[10],AnmaxTc[11],AnmaxTc[12],AnmaxTc[13],AnmaxTc[14],AnmaxTc[15],AnmaxTc[16],AnmaxTc[17],\
    AnmaxTc[18],AnmaxTc[19],AnmaxTc[20],AnmaxTc[21],AnmaxTc[22],AnmaxTc[23],AnmaxTc[24],AnmaxTc[25],AnmaxTc[26],AnmaxTc[27],\
    AnmaxTc[28],AnmaxTc[29],AnmaxTc[30],AnmaxTc[31],AnmaxTc[32],AnmaxTc[33],AnmaxTc[34],AnmaxTc[35],AnmaxTc[36],AnmaxTc[37],\
    AnmaxTc[38],AnmaxTc[39],AnmaxTc[40],AnmaxTc[41],AnmaxTc[42],AnmaxTc[43],AnmaxTc[44],AnmaxTc[45],AnmaxTc[46],AnmaxTc[47],\
    AnmaxTc[48]) )
    
    # --- write output for avgTc
    favgTcstat.write( ('%10d%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f\n') \
    % (int(daydirM1s[kmk]),AnavgTc[0], AnavgTc[1],AnavgTc[2],AnavgTc[3],AnavgTc[4],AnavgTc[5],AnavgTc[6],AnavgTc[7],\
    AnavgTc [8],AnavgTc [9],AnavgTc[10],AnavgTc[11],AnavgTc[12],AnavgTc[13],AnavgTc[14],AnavgTc[15],AnavgTc[16],AnavgTc[17],\
    AnavgTc[18],AnavgTc[19],AnavgTc[20],AnavgTc[21],AnavgTc[22],AnavgTc[23],AnavgTc[24],AnavgTc[25],AnavgTc[26],AnavgTc[27],\
    AnavgTc[28],AnavgTc[29],AnavgTc[30],AnavgTc[31],AnavgTc[32],AnavgTc[33],AnavgTc[34],AnavgTc[35],AnavgTc[36],AnavgTc[37],\
    AnavgTc[38],AnavgTc[39],AnavgTc[40],AnavgTc[41],AnavgTc[42],AnavgTc[43],AnavgTc[44],AnavgTc[45],AnavgTc[46],AnavgTc[47],\
    AnavgTc[48]) )

    # --- write output for MAEs
    fmaestat.write( ('%10d%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f\n') \
    % (int(daydirM1s[kmk]),difMAEfl[0], difMAEfl[1],difMAEfl[2],difMAEfl[3],difMAEfl[4],difMAEfl[5],difMAEfl[6],difMAEfl[7],\
    difMAEfl [8],difMAEfl [9],difMAEfl[10],difMAEfl[11],difMAEfl[12],difMAEfl[13],difMAEfl[14],difMAEfl[15],difMAEfl[16],difMAEfl[17],\
    difMAEfl[18],difMAEfl[19],difMAEfl[20],difMAEfl[21],difMAEfl[22],difMAEfl[23],difMAEfl[24],difMAEfl[25],difMAEfl[26],difMAEfl[27],\
    difMAEfl[28],difMAEfl[29],difMAEfl[30],difMAEfl[31],difMAEfl[32],difMAEfl[33],difMAEfl[34],difMAEfl[35],difMAEfl[36],difMAEfl[37],\
    difMAEfl[38],difMAEfl[39],difMAEfl[40],difMAEfl[41],difMAEfl[42],difMAEfl[43],difMAEfl[44],difMAEfl[45],difMAEfl[46],difMAEfl[47],\
    difMAEfl[48]) )

    # --- write output for Hit-Rates  
    fhitrstat.write( ('%10d%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f%7.2f\n') \
    % (int(daydirM1s[kmk]),difHRfl[0], difHRfl[1],difHRfl[2],difHRfl[3],difHRfl[4],difHRfl[5],difHRfl[6],difHRfl[7],\
    difHRfl [8],difHRfl [9],difHRfl[10],difHRfl[11],difHRfl[12],difHRfl[13],difHRfl[14],difHRfl[15],difHRfl[16],difHRfl[17],\
    difHRfl[18],difHRfl[19],difHRfl[20],difHRfl[21],difHRfl[22],difHRfl[23],difHRfl[24],difHRfl[25],difHRfl[26],difHRfl[27],\
    difHRfl[28],difHRfl[29],difHRfl[30],difHRfl[31],difHRfl[32],difHRfl[33],difHRfl[34],difHRfl[35],difHRfl[36],difHRfl[37],\
    difHRfl[38],difHRfl[39],difHRfl[40],difHRfl[41],difHRfl[42],difHRfl[43],difHRfl[44],difHRfl[45],difHRfl[46],difHRfl[47],\
    difHRfl[48]) )
    
    #t print 'MDbiasST =', MDbiasST 
    #t for ii in range (0,nptsMD):
    #t    print 'ii, MDidst, MDndays, MDnGFL, MDprGF, MDbiasST (00-48) =', (('%3d%9d%4d%4d%4d%2d%2d') % \
    #t    (ii+1, MDidst[ii], MDndays[ii], MDnGFL[ii], MDprGFL[ii], MDbiasST[ii,0], MDbiasST[ii,48] ))
    #t sys.exit()

print '===== STATISTICS OVER THE PERIOD: ', cMETPAR
print 'BIAS     --- gAnsignBIAS, bAnsignBIAS =', gAnsignBIAS, bAnsignBIAS
print 'MAE      --- gAnsignMAE,  bAnsignMAE  =', gAnsignMAE,  bAnsignMAE
print 'Hit-Rate --- gAnsignHR,   bAnsignHR   =', gAnsignHR,   bAnsignHR


# ----- CLOSING FILES WITH WRITTEN OUTPUT -------------------------------------
fokstat.close()    # --- with perecentage of OK-corrected cases
fbiasostat.close() # --- with old bias
fbiasnstat.close() # --- with new bias
fmaxTcstat.close() # --- with maximum correction
fminTcstat.close() # --- with minimum correction
favgTcstat.close() # --- with average correction
fhitrstat.close()  # --- with hit-rate
fmaestat.close()   # --- with MAE


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                          PLOTTING SAVED OUTPUT
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
xxd = NFLen
FLenticks = ['00',\
'01','02','03','04','05','06','07','08','09','10',\
'11','12','13','14','15','16','17','18','19','20',\
'21','22','23','24','25','26','27','28','29','30',\
'31','32','33','34','35','36','37','38','39','40',\
'41','42','43','44','45','46','47','48']

# =============================================================================
# --1-- PLOT : MAES BY DAYS vs FOREACST LENGTHS
# =============================================================================
# --- PLOT MATRIX of MAEs improvements as a function of days within a month 
#     for specified UTC run

print ' fmaefile-file : ', fmaefile

# ----- extracting ONLY necessary data
dataFM = np.genfromtxt(fmaefile, dtype = float, skip_header=1)
#t print 'dataFM[:] =', dataFM
ndates = dataFM.shape[0]
print 'ndates =', ndates
dataMM = dataFM[:, 1:] # -- reshape matrix keeping only required data
#t print 'dataMM[:] =', dataMM
dataAA = np.genfromtxt(fmaefile, dtype = str, skip_header=1)
dataMMDD = dataAA[:, 0] # -- reshape matrix keeping only YYYYMMDDHH data
#t print 'dataMMDD[:] =', dataMMDD 

#A TdataMMDD = np.chararray(ndates,14)
#A for ij in range (0, ndates):
#A    TdataMMDD[ij] = dataMMDD[ij] + '  ' + aNncm
#A    print TdataMMDD[ij]
    
    
fig = plt.figure(); plt.clf()

plt.imshow(dataMM, cmap=plt.cm.hot_r, interpolation='nearest') # jet

if iflagMETPAR == 1: # --- T2m
    cb=plt.colorbar(ticks=[-0.5,-0.45,-0.4,-0.35,-0.3,-0.25,-0.2,-0.15,-0.1,-0.05,\
    0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5], shrink=0.8, orientation='horizontal')
if iflagMETPAR == 2: # --- W10m
    cb=plt.colorbar(ticks=[-1.0,-0.8, -0.6, -0.4, -0.2, 0., 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2], shrink=0.8, orientation='horizontal')
    
#cb.ax.set_xticklabels(['-0.5','-0.45','-0.4','-0.35','-0.3','-0.25','-0.2','-0.15','-0.1','-0.05',\
#'0', '0.05', '0.1', '0.15', '0.2', '0.25', '0.3', '0.35', '0.4', '0.45', '0.5'],\
#fontsize=10, rotation=90,)
textMETPAR = 'Change of MAE ' + cMPUNITS
cb.set_label(textMETPAR, fontsize=10)

Ttitle = textYY + '-' + textMM + ' : ' + cMETPAR  + ' : Differences in MAEs : '+ cMODEL +' model runs at ' + textUTChh + ' UTC'
plt.title(Ttitle, fontsize=12)
pxx = NFLen/xxd ; pyy = ndates + 4
Ttext = ainfo + ' : Forecast Lengths (hours) : ' + textUTChh + ' UTC + FL'
plt.text(pxx, pyy, Ttext, style='italic', fontsize=12)
        
plt.xticks(np.arange(0,49), FLenticks, rotation=90, fontsize=8)       
plt.yticks(np.arange(0,ndates), dataMMDD, horizontalalignment='right', fontsize=8)  

filefig = 'fig'+ cMODEL +'_maes_' + cMETPAR + '_' + cinfo + '_' + textYY + textMM + '_' + textUTChh + 'utc.png'
plt.savefig(filefig, format='png')


# =============================================================================
# --2-- PLOT : OLD BIAS BY DAYS vs FOREACST LENGTHS
# =============================================================================
# --- PLOT MATRIX of BIAS OLD as a function of days within a month 
#     for specified UTC run

print ' fbiasofile-file : ', fbiasofile

# ----- extracting ONLY necessary data
dataFM = np.genfromtxt(fbiasofile, dtype = float, skip_header=1)
#t print 'dataFM[:] =', dataFM
ndates = dataFM.shape[0]
print 'ndates =', ndates
dataMM = dataFM[:, 1:] # -- reshape matrix keeping only required data
#t print 'dataMM[:] =', dataMM
dataAA = np.genfromtxt(fbiasofile, dtype = str, skip_header=1)
dataMMDD = dataAA[:, 0] # -- reshape matrix keeping only YYYYMMDDHH data
#t print 'dataMMDD[:] =', dataMMDD 


fig = plt.figure(); plt.clf()

plt.imshow(dataMM, cmap=plt.cm.jet, interpolation='nearest') # jet, hot_r

if iflagMETPAR == 1: # --- T2m
    cb=plt.colorbar(ticks=[-1.5,-1.4,-1.3,-1.2,-1.1,-1,-0.9,-0.8,-0.7,-0.6,-0.5,\
    -0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.1,1.2,1.3,1.4,1.5],\
    shrink=0.8, orientation='horizontal')
if iflagMETPAR == 2: # --- W10m
    cb=plt.colorbar(ticks=[-5., -4.5, -4.,-3.5, -3., -2.5, -2., -1.5, -1., -0.5, 0., 0.5, 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5.],\
    shrink=0.8, orientation='horizontal')
    
#cb.ax.set_xticklabels(['<-1.5','-1.5','-1.4','-1.3','-1.2','-1.1','-1','-0.9','-0.8',\
#'-0.7','-0.6','-0.5','-0.4','-0.3','-0.2','-0.1','0','0.1','0.2','0.3','0.4',\
#'0.5','0.6','0.7','0.8','0.9','1.1','1.2','1.3','1.4','1.5','>1.5'],\
#fontsize=10, rotation=90,)
textMETPAR = 'Change of Old Bias ' + cMPUNITS
cb.set_label(textMETPAR, fontsize=10)

Ttitle = textYY + '-' + textMM + ' : ' + cMETPAR  + ' : Old Bias : '+ cMODEL +' model runs at ' + textUTChh + ' UTC'
plt.title(Ttitle, fontsize=12)
pxx = NFLen/xxd ; pyy = ndates + 4
Ttext = ainfo + ' : Forecast Lengths (hours) : ' + textUTChh + ' UTC + FL'
plt.text(pxx, pyy, Ttext, style='italic', fontsize=12)
        
plt.xticks(np.arange(0,49), FLenticks, rotation=90, fontsize=8)      
plt.yticks(np.arange(0,ndates), dataMMDD, horizontalalignment='right', fontsize=8)  

filefig = 'fig_'+ cMODEL +'_biaso_' + cMETPAR + '_' + cinfo + '_' + textYY + textMM + '_' + textUTChh + 'utc.png'
plt.savefig(filefig, format='png')


# =============================================================================
# --3-- PLOT : NEW BIAS BY DAYS vs FOREACST LENGTHS
# =============================================================================
# --- PLOT MATRIX of BIAS NEW as a function of days within a month 
#     for specified UTC run

print ' fbiasnfile-file : ', fbiasnfile

# ----- extracting ONLY necessary data
dataFM = np.genfromtxt(fbiasnfile, dtype = float, skip_header=1)
#t print 'dataFM[:] =', dataFM
ndates = dataFM.shape[0]
print 'ndates =', ndates
dataMM = dataFM[:, 1:] # -- reshape matrix keeping only required data
#t print 'dataMM[:] =', dataMM
dataAA = np.genfromtxt(fbiasnfile, dtype = str, skip_header=1)
dataMMDD = dataAA[:, 0] # -- reshape matrix keeping only YYYYMMDDHH data
#t print 'dataMMDD[:] =', dataMMDD 


fig = plt.figure(); plt.clf()

plt.imshow(dataMM, cmap=plt.cm.jet, interpolation='nearest') # jet, hot_r

if iflagMETPAR == 1: # --- T2m
    cb=plt.colorbar(ticks=[-1.5,-1.4,-1.3,-1.2,-1.1,-1,-0.9,-0.8,-0.7,-0.6,-0.5,\
    -0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.1,1.2,1.3,1.4,1.5],\
    shrink=0.8, orientation='horizontal')
if iflagMETPAR == 2: # --- W10m
    cb=plt.colorbar(ticks=[-5., -4.5, -4.,-3.5,-3.,-2.5, -2., -1.5, -1., -0.5, 0., 0.5, 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5.],\
    shrink=0.8, orientation='horizontal')

#cb.ax.set_xticklabels(['<-1.5','-1.5','-1.4','-1.3','-1.2','-1.1','-1','-0.9','-0.8',\
#'-0.7','-0.6','-0.5','-0.4','-0.3','-0.2','-0.1','0','0.1','0.2','0.3','0.4',\
#'0.5','0.6','0.7','0.8','0.9','1.1','1.2','1.3','1.4','1.5','>1.5'],\
#fontsize=10, rotation=90,)
textMETPAR = 'Change of New Bias ' + cMPUNITS
cb.set_label(textMETPAR, fontsize=10)

Ttitle = textYY + '-' + textMM + ' : ' + cMETPAR  + ' : New Bias : '+ cMODEL +' model runs at ' + textUTChh + ' UTC'
plt.title(Ttitle, fontsize=12)
pxx = NFLen/xxd ; pyy = ndates + 4
Ttext = ainfo + ' : Forecast Lengths (hours) : ' + textUTChh + ' UTC + FL'
plt.text(pxx, pyy, Ttext, style='italic', fontsize=12)
        
plt.xticks(np.arange(0,49), FLenticks, rotation=90, fontsize=8)      
plt.yticks(np.arange(0,ndates), dataMMDD, horizontalalignment='right', fontsize=8)  

filefig = 'fig_'+ cMODEL +'_biasn_' + cMETPAR + '_' + cinfo + '_' + textYY + textMM + '_' + textUTChh + 'utc.png'
plt.savefig(filefig, format='png')


# =============================================================================
# --4-- PLOT : HIT-RATES BY DAYS vs FOREACST LENGTHS
# =============================================================================
# --- PLOT MATRIX of Hit-Rates improvements as a function of days within a month
#     for specified UTC runs
      
print ' fhitrfile-file : ', fhitrfile


# ----- extracting ONLY necessary data
dataFM = np.genfromtxt(fhitrfile, dtype = float, skip_header=1)
#t print 'dataFM[:] =', dataFM
ndates = dataFM.shape[0]
print 'ndates =', ndates
dataMM = dataFM[:, 1:] # -- reshape matrix keeping only required data
#t print 'dataMM[:] =', dataMM
dataAA = np.genfromtxt(fhitrfile, dtype = str, skip_header=1)
dataMMDD = dataAA[:, 0] # -- reshape matrix keeping only YYYYMMDDHH data
#t print 'dataMMDD[:] =', dataMMDD 


fig = plt.figure(); plt.clf()

plt.imshow(dataMM, cmap=plt.cm.hot_r, interpolation='nearest') # jet

if iflagMETPAR == 1: # --- T2m
    cb=plt.colorbar(ticks=[-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,\
    11,12,13,14,15], shrink=0.8, orientation='horizontal')
if iflagMETPAR == 2: # --- W10m
    cb=plt.colorbar(ticks=[-10,-8,-6.,-4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30],\
    shrink=0.8, orientation='horizontal')

#cb.ax.set_xticklabels(['<-10','-10','-9','-8','-7','-6','-5','-4','-3','-2','-1','0',\
#'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','>15'],\
#fontsize=10, rotation=90,)
textMETPAR = 'Change of Hit-Rate [%]'
cb.set_label(textMETPAR, fontsize=10)

Ttitle = textYY + '-' + textMM + ' : ' + cMETPAR  + ' : Differences in Hit-Rates : '+ cMODEL +' model runs at ' + textUTChh + ' UTC'
plt.title(Ttitle, fontsize=12)
pxx = NFLen/xxd ; pyy = ndates + 4
Ttext = ainfo + ' : Forecast Lengths (hours) : ' + textUTChh + ' UTC + FL'
plt.text(pxx, pyy, Ttext, style='italic', fontsize=12)
        
plt.xticks(np.arange(0,49), FLenticks, rotation=90, fontsize=8) 
plt.yticks(np.arange(0,ndates), dataMMDD, horizontalalignment='right', fontsize=8)  

filefig = 'fig_'+ cMODEL +'_hitrate_' + cMETPAR + '_' + cinfo + '_' + textYY + textMM + '_' + textUTChh + 'utc.png'  
plt.savefig(filefig, format='png')     
           
           

# =============================================================================
# --5-- PLOT : avgTc BY DAYS vs FOREACST LENGTHS
# =============================================================================
       
print ' favgTcfile-file : ', favgTcfile

# ----- extracting ONLY necessary data
dataFM = np.genfromtxt(favgTcfile, dtype = float, skip_header=1)
#t print 'dataFM[:] =', dataFM
ndates = dataFM.shape[0]
print 'ndates =', ndates
dataMM = dataFM[:, 1:] # -- reshape matrix keeping only required data
#t print 'dataMM[:] =', dataMM
dataAA = np.genfromtxt(favgTcfile, dtype = str, skip_header=1)
dataMMDD = dataAA[:, 0] # -- reshape matrix keeping only YYYYMMDDHH data
#t print 'dataMMDD[:] =', dataMMDD 

fig = plt.figure(); plt.clf()

plt.imshow(dataMM, cmap=plt.cm.hot_r, interpolation='nearest') # jet

if iflagMETPAR == 1: # --- T2m
    cb=plt.colorbar(ticks=[-1.5,-1.4,-1.3,-1.2,-1.1,-1,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,\
    -0.1,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5],\
    shrink=0.8, orientation='horizontal')
if iflagMETPAR == 2: # --- W10m
    cb=plt.colorbar(ticks=[-2,-1.8,-1.6,-1.4, -1.2, -1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2],\
    shrink=0.8, orientation='horizontal')

#cb.ax.set_xticklabels(['<-1.5','-1.5','-1.4','-1.3','-1.2','-1.1','-1','-0.9',\
#'-0.8','-0.7','-0.6','-0.5','-0.4','-0.3','-0.2','-0.1','0','0.1','0.2','0.3',\
#'0.4','0.5','0.6','0.7','0.8','0.9','1','1.1','1.2','1.3','1.4','1.5','>1.5'],\
#fontsize=10, rotation=90,)
textMETPAR = 'Average correction to ' + cMETPAR + ' ' + cMPUNITS
cb.set_label(textMETPAR, fontsize=10)

Ttitle = textYY + '-' + textMM + ' : ' + cMETPAR  + ' : Average correction : '+ cMODEL +' model runs at ' + textUTChh + ' UTC'
plt.title(Ttitle, fontsize=12)
pxx = NFLen/xxd ; pyy = ndates + 4
Ttext = ainfo + ' : Forecast Lengths (hours) : ' + textUTChh + ' UTC + FL'
plt.text(pxx, pyy, Ttext, style='italic', fontsize=12)
        
plt.xticks(np.arange(0,49), FLenticks, rotation=90, fontsize=8)      
plt.yticks(np.arange(0,ndates), dataMMDD, horizontalalignment='right', fontsize=8)  

filefig = 'fig_'+ cMODEL +'_avg_' + cMETPAR + '_' + cinfo + '_' + textYY + textMM + '_' + textUTChh + 'utc.png'
plt.savefig(filefig, format='png')



# =============================================================================
# --6-- PLOT : minTc BY DAYS vs FOREACST LENGTHS
# =============================================================================
       
print ' fminTcfile-file : ', fminTcfile

# ----- extracting ONLY necessary data
dataFM = np.genfromtxt(fminTcfile, dtype = float, skip_header=1)
#t print 'dataFM[:] =', dataFM
ndates = dataFM.shape[0]
print 'ndates =', ndates
dataMM = dataFM[:, 1:] # -- reshape matrix keeping only required data
#t print 'dataMM[:] =', dataMM
dataAA = np.genfromtxt(fminTcfile, dtype = str, skip_header=1)
dataMMDD = dataAA[:, 0] # -- reshape matrix keeping only YYYYMMDDHH data
#t print 'dataMMDD[:] =', dataMMDD 


fig = plt.figure(); plt.clf()

plt.imshow(dataMM, cmap=plt.cm.Blues_r, interpolation='nearest') # jet

if iflagMETPAR == 1: # --- T2m
    cb=plt.colorbar(ticks=[-1.5,-1.4,-1.3,-1.2,-1.1,-1,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2],\
    shrink=0.8, orientation='horizontal')
if iflagMETPAR == 2: # --- W10m
    cb=plt.colorbar(ticks=[-4, -3.8, -3.6, -3.4, -3.2, -3,-2.8,-2.6,-2.4,-2.2,-2,-1.8,-1.6,-1.4, -1.2, -1, -0.8, -0.6, -0.4, -0.2, 0],\
    shrink=0.8, orientation='horizontal')

#cb.ax.set_xticklabels(['<-1.5','-1.5','-1.4','-1.3','-1.2','-1.1','-1','-0.9',\
#'-0.8','-0.7','-0.6','-0.5','-0.4','-0.3','-0.2','-0.1','0','0.1','0.2'],\
#fontsize=10, rotation=90,)
textMETPAR = 'Maximum negative correction to ' + cMETPAR + ' ' + cMPUNITS
cb.set_label(textMETPAR, fontsize=10)

Ttitle = textYY + '-' + textMM + ' : ' + cMETPAR  + ' : Max negative correction : '+ cMODEL +' model runs at ' + textUTChh + ' UTC'
plt.title(Ttitle, fontsize=12)
pxx = NFLen/xxd ; pyy = ndates + 4
Ttext = ainfo + ' : Forecast Lengths (hours) : ' + textUTChh + ' UTC + FL'
plt.text(pxx, pyy, Ttext, style='italic', fontsize=12)
        
plt.xticks(np.arange(0,49), FLenticks, rotation=90, fontsize=8)     
plt.yticks(np.arange(0,ndates), dataMMDD, horizontalalignment='right', fontsize=8)  

filefig = 'fig_'+ cMODEL +'_min_' + cMETPAR + '_' + cinfo + '_' + textYY + textMM + '_' + textUTChh + 'utc.png'
plt.savefig(filefig, format='png')


# =============================================================================
# --7-- PLOT : maxTc BY DAYS vs FOREACST LENGTHS
# =============================================================================
    
print ' fmaxTcfile-file : ', fmaxTcfile

# ----- extracting ONLY necessary data
dataFM = np.genfromtxt(fmaxTcfile, dtype = float, skip_header=1)
#t print 'dataFM[:] =', dataFM
ndates = dataFM.shape[0]
print 'ndates =', ndates
dataMM = dataFM[:, 1:] # -- reshape matrix keeping only required data
#t print 'dataMM[:] =', dataMM
dataAA = np.genfromtxt(fmaxTcfile, dtype = str, skip_header=1)
dataMMDD = dataAA[:, 0] # -- reshape matrix keeping only YYYYMMDDHH data
#t print 'dataMMDD[:] =', dataMMDD 

 
fig = plt.figure(); plt.clf()

plt.imshow(dataMM, cmap=plt.cm.hot_r, interpolation='nearest') # hot_r, autumn_r

if iflagMETPAR == 1: # --- T2m
    cb=plt.colorbar(ticks=[-0.1,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,\
    1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5],\
    shrink=0.8, orientation='horizontal')
if iflagMETPAR == 2: # --- W10m
    cb=plt.colorbar(ticks=[0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3, 3.2, 3.4, 3.6, 3.8, 4],\
    shrink=0.8, orientation='horizontal')

#cb.ax.set_xticklabels(['-0.1','0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8',\
#'0.9','1','1.1','1.2','1.3','1.4','1.5','1.6','1.7','1.8','1.9','2.0','2.1','2.2',\
#'2.3','2.4','2.5','>2.5'],\
#fontsize=10, rotation=90,)
textMETPAR = 'Maximum positive correction to ' + cMETPAR + ' ' + cMPUNITS
cb.set_label(textMETPAR, fontsize=10)

Ttitle = textYY + '-' + textMM + ' : ' + cMETPAR  + ' : Max positive correction : '+ cMODEL +' model runs at ' + textUTChh + ' UTC'
plt.title(Ttitle, fontsize=12)
pxx = NFLen/xxd ; pyy = ndates + 4
Ttext = ainfo + ' : Forecast Lengths (hours) : ' + textUTChh + ' UTC + FL'
plt.text(pxx, pyy, Ttext, style='italic', fontsize=12)
        
plt.xticks(np.arange(0,49), FLenticks, rotation=90, fontsize=8)      
plt.yticks(np.arange(0,ndates), dataMMDD, horizontalalignment='right', fontsize=8)  

filefig = 'fig_'+ cMODEL +'_max_' + cMETPAR + '_' + cinfo + '_' + textYY + textMM + '_' + textUTChh + 'utc.png'
plt.savefig(filefig, format='png')



# =============================================================================
# --8-- PLOT : %-OK CORRECTED CASES BY DAYS vs FOREACST LENGTHS
# =============================================================================
    
print ' fokfile-file : ', fokfile

# ----- extracting ONLY necessary data
dataFM = np.genfromtxt(fokfile, dtype = float, skip_header=1)
#t print 'dataFM[:] =', dataFM
ndates = dataFM.shape[0]
print 'ndates =', ndates
dataMM = dataFM[:, 1:] # -- reshape matrix keeping only required data
#t print 'dataMM[:] =', dataMM
dataAA = np.genfromtxt(fokfile, dtype = str, skip_header=1)
dataMMDD = dataAA[:, 0] # -- reshape matrix keeping only YYYYMMDDHH data
#t print 'dataMMDD[:] =', dataMMDD 

 
fig = plt.figure(); plt.clf()

plt.imshow(dataMM, cmap=plt.cm.gist_rainbow_r, interpolation='nearest') # hot_r, autumn_r
cb=plt.colorbar(ticks=[0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105],\
shrink=0.8, orientation='horizontal')

#cb.ax.set_xticklabels(['0','5','10','15','20','25','30','35','40','45','50','55',\
#'60','65','70','75','80','85','90','95','100','105'],\
#fontsize=10, rotation=90,)
cb.set_label('Corrected cases [%]', fontsize=10)

Ttitle = textYY + '-' + textMM + ' : ' + cMETPAR  + ' : Percentage of corrected cases : '+ cMODEL +' model runs at ' + textUTChh + ' UTC'
plt.title(Ttitle, fontsize=12)
pxx = NFLen/xxd ; pyy = ndates + 4
Ttext = ainfo + ' : Forecast Lengths (hours) : ' + textUTChh + ' UTC + FL'
plt.text(pxx, pyy, Ttext, style='italic', fontsize=12)
        
plt.xticks(np.arange(0,49), FLenticks, rotation=90, fontsize=8)      
plt.yticks(np.arange(0,ndates), dataMMDD, horizontalalignment='right', fontsize=8)  

filefig = 'fig_'+ cMODEL +'_ok_' + cMETPAR + '_' + cinfo + '_' + textYY + textMM + '_' + textUTChh + 'utc.png'
plt.savefig(filefig, format='png')

# =============================================================================

# =============================================================================

  
sys.exit()

