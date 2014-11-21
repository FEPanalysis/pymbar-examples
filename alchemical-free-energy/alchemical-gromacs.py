#!/bin/env python

# Originally written by Michael Shirts as:
# Example illustrating the use of MBAR for computing the hydration free energy of OPLS 3-methylindole
# in TIP3P water through alchemical free energy simulations.

# Adapted by P. Klimovich and D. Mobley, March 2011, to be slightly more general.
# Additionally adapted by Michael Shirts and P. Klimovich, May 2013

#===================================================================================================
# IMPORTS
#===================================================================================================

import pymbar     ## Multistate Bennett Acceptance Ratio estimator ##
import timeseries ## for timeseries analysis                       ##

import numpy
import pickle
import re # for regular expressions
from glob import glob # for pathname matching
from collections import Counter # for counting elements in an array
from optparse import OptionParser # for parsing command-line options
import os # operating system dependent modules of Python
import time as ttt_time # for timing
import matplotlib # for making plots, version 'matplotlib-1.1.0-1'; errors may pop up when using earlier versions
matplotlib.use('Agg')
import matplotlib.pyplot as pl
from matplotlib.font_manager import FontProperties as FP

import pdb # for debugging

#===================================================================================================
# CONSTANTS AND PARAMETERS
#===================================================================================================

kB = 1.381*6.02214/1000.0 # Boltzmann's constant (kJ/mol/K).
relative_tolerance = 1e-10 # Convergence criterion of the energy estimates for BAR and MBAR.
methods = ['TI','TI-CUBIC','DEXP','IEXP','BAR','MBAR'] # Free energy estimation methods.
#methods = ['TI','TI-CUBIC','DEXP','IEXP','GINS','GDEL','BAR','UBAR','RBAR','MBAR'] # Full list of supported estimators.

#===================================================================================================
# INPUT OPTIONS
#===================================================================================================

parser = OptionParser()
parser.add_option('-a', '--software', dest = 'software', help = 'Package\'s name data files come from. Default: Gromacs.', default = 'Gromacs')
parser.add_option('-d', '--dir', dest = 'datafile_directory', help = 'Directory in which data files are stored. Default: Current directory.', default = '.')
parser.add_option('-f', '--forwrev', dest = 'bForwrev', help = 'Plotting the free energy change as a function of time in both directions. The number of time points (an integer) is to be followed the flag. Default: 0', default = 0, type=int)
parser.add_option('-g', '--breakdown', dest = 'breakdown', help = 'Plotting the free energy differences evaluated for each pair of adjacent states for all methods. Default: False.', default = False, action = 'store_true')
parser.add_option('-i', '--threshold', dest = 'uncorr_threshold', help = 'Perform the analysis with rather all the data if the number of uncorrelated samples is found to be less than this number.', default = 50, type=int)
parser.add_option('-k', '--koff', dest = 'bSkipLambdaIndex', help = 'Give a string of lambda indices separated by \'-\' and they will be removed from the analysis. (Another approach is to have only the files of interest present in the directory). Default: None.', default = '')
parser.add_option('-o', '--out', dest = 'output_directory', help = 'Directory in which the output files produced by this script will be stored. Default: Same as datafile_directory.', default = '')
parser.add_option('-p', '--prefix', dest = 'prefix', help = 'Prefix for datafile sets, i.e.\'prod\' (default).', default = 'prod')
parser.add_option('-q', '--suffix', dest = 'suffix', help = 'Suffix for datafile sets, i.e. \'xvg\' (default).', default = 'xvg')
parser.add_option('-s', '--skiptime', dest = 'equiltime', help = 'Discard data prior to this specified time as \'equilibration\' data. Units picoseconds. Default: 100 ps.', default = 100, type=float)
parser.add_option('-t', '--temperature', dest = 'temperature', help = "Temperature in K. Default: 298 K.", default = 298, type=float)
parser.add_option('-u', '--units', dest = 'units', help = 'Units to report energies: \'kJ\', \'kcal\', and \'kBT\'. Default: \'kJ\'', default = 'kJ')
parser.add_option('-v', '--verbose', dest = 'verbose', help = 'Verbose option for BAR and MBAR. Default: False.', default = False, action = 'store_true')
parser.add_option('-w', '--overlap', dest = 'overlap', help = 'Print out and plot the overlap matrix. Default: False.', default = False, action = 'store_true')
parser.add_option('-x', '--ignoreWL', dest = 'bIgnoreWL', help = 'Do not check whether the WL weights are equilibrated. No log file needed as an accompanying input.', default = False, action = 'store_true')
parser.add_option('-z', '--initialize', dest = 'init_with', help = 'The initial MBAR free energy guess; either \'BAR\' or \'zeroes\'. Default: \'BAR\'.', default = 'BAR')

#===================================================================================================
# FUNCTIONS: The unix-like helpers.
#===================================================================================================

def wcPy(f):
   """Count up lines in file 'f'."""
   if not type(f) is file:
      with open(f, 'r') as f:
         return wcPy(f)
   return sum(1 for l in f)

def trPy(l, s='[,\\\\"/()-]', char=' '):
   """In string 'l' replace all the charachters from 's' with 'char'."""
   return re.sub(s, char, l)

def grepPy(f, s):
   """From file 'f' extract the (first occurence of) line that contains string 's'."""
   if not type(f) is file:
      with open(f, 'r') as f:
         return grepPy(f, s)
   for line in f:
      if s in line:
         return line
   return ''

def tailPy(f, nlines, lenb=1024):
   if not type(f) is file:
      with open(f, 'r') as f:
         return tailPy(f, nlines, lenb)
   f.seek(0, 2)
   sizeb = f.tell()
   n_togo = nlines
   i = 1
   excerpt = []
   while n_togo > 0 and sizeb > 0:
      if (sizeb - lenb > 0):
         f.seek(-i*lenb, 2)
         excerpt.append(f.read(lenb))
      else:
         f.seek(0,0)
         excerpt.append(f.read(sizeb))
      ll = excerpt[-1].count('\n')
      n_togo -= ll
      sizeb -= lenb
      i += 1
   return ''.join(excerpt).splitlines()[-nlines:]

#===================================================================================================
# FUNCTIONS: This is the Gromacs dhdl.xvg file parser.
#===================================================================================================
   
def readDataGromacs():
   """Read in .xvg files; return nsnapshots, lv, dhdlt, and u_klt."""
   
   class F:
      """This is the object to be built on the filename."""
 
      def __init__(self, filename):
         self.filename = filename
 
      def sortedHelper(self):
         """This function will assist the built-in 'sorted' to sort filenames.
            Returns a tuple whose first element is an integer while others are strings."""
         meat = os.path.basename(self.filename).replace(P.prefix, '').replace(P.suffix, '')
         l = [i for i in re.split('\.|-|_', meat) if i]
         try:
            self.state = l[0] = int(l[0]) # Will be of use for selective MBAR analysis.
         except ValueError:
            parser.error("\nFile's prefix should be followed by a numerical character. Cannot sort the files.")
         return tuple(l)
 
      def readHeader(self):
         self.skip_lines   = 0  # Number of lines from the top that are to be skipped.
         self.lv_names     = () # Lambda type names, e.g. 'coul', 'vdw'.
         snap_size         = [] # Time from first two snapshots to determine snapshot's size.
         self.lv           = [] # Lambda vectors, e.g. (0, 0), (0.2, 0), (0.5, 0).
 
         self.bEnergy   = False
         self.bPV       = False
         self.bExpanded = False
 
         print "Reading metadata from %s..." % self.filename
         with open(self.filename,'r') as infile:
            for line in infile:
 
               if line.startswith('#'):
                  self.skip_lines += 1
 
               elif line.startswith('@'):
                  self.skip_lines += 1
                  elements = trPy(line).split()
                  if not 'legend' in elements:
                     continue
 
                  if 'Energy' in elements:
                     self.bEnergy = True
                  if 'pV' in elements:
                     self.bPV = True
                  if 'state' in elements:
                     self.bExpanded = True
 
                  if 'dH' in elements:
                     self.lv_names += elements[7],
                  if 'xD' in elements:
                     self.lv.append(elements[-len(self.lv_names):])
 
               else:
                  snap_size.append(float(line.split()[0]))
                  if len(snap_size) > 1:
                     self.snap_size = numpy.diff(snap_size)[0]
                     P.snap_size.append(self.snap_size)
                     break
         return self.lv
 
      def iter_loadtxt(self, state):
         """Houstonian Joe Kington claims it is faster than numpy.loadtxt:
         http://stackoverflow.com/questions/8956832/python-out-of-memory-on-large-csv-file-numpy"""
         
         def iter_func():
            with open(self.filename, 'r') as infile:
               for _ in range(self.skip_lines):
                  next(infile)
               for line in infile:
                  line = line.split()
                  for item in line:
                     yield item
 
         def slice_data(data, state=state):
            # Where the dE columns should be stored.
            if (len(ndE_unique)>1 and ndE[state]<4):
               # If BAR, store shifted 2/3 arrays.
               s1, s2 = numpy.array((0, ndE[state])) + state-(state>0)
            else:
               # If MBAR or selective MBAR or BAR/MBAR, store all.
               s1, s2 = (0, K)
            # Which dhdl columns are to be read.
            read_dhdl_sta = 1+self.bEnergy+self.bExpanded
            read_dhdl_end = read_dhdl_sta + n_components
  
            data = data.T
            dhdlt[state, :, :nsnapshots[state]] = data[read_dhdl_sta : read_dhdl_end, :]
  
            if not bSelective_MBAR:
               r1, r2 = (read_dhdl_end, read_dhdl_end+K)
               if bPV:
                  u_klt[state, s1:s2, :nsnapshots[state]] = beta * ( data[r1:r2, :] + data[-1,:] )
               else:
                  u_klt[state, s1:s2, :nsnapshots[state]] = beta * data[r1:r2, :]
            else: # can't do slicing; prepare a mask (slicing is thought to be faster/less memory consuming than masking)
               mask_read_uklt = numpy.array( [0]*read_dhdl_end + [1 if (k in sel_states) else 0 for k in range(ndE[0])] + ([0] if bPV else []), bool )
               if bPV:
                  u_klt[state, s1:s2, :nsnapshots[state]] = beta * ( data[mask_read_uklt, :] + data[-1,:] )
               else:
                  u_klt[state, s1:s2, :nsnapshots[state]] = beta * data[mask_read_uklt, :]
            return
 
         print "Loading in data from %s (%s) ..." % (self.filename, "all states" if self.bExpanded else 'state %d' % state)
         data = numpy.fromiter(iter_func(), dtype=float)
         if not self.len_first == self.len_last:
            data = data[: -self.len_last]
         data = data.reshape((-1, self.len_first))
 
         if self.bExpanded:
            for k in range(K):
               mask_k = (data[:, 1] == k)
               data_k = data[mask_k]
               slice_data(data_k, k)
         else:
            slice_data(data)
 
      def parseLog(self):
         """By parsing the .log file of the expanded-ensemble simulation
         find out the time in ps when the WL equilibration has been reached.
         Return the greater of WLequiltime and equiltime."""
         if not(P.bIgnoreWL):
            logfilename = self.filename.replace('.xvg', '.log')
            if not os.path.isfile(logfilename):
               parser.error("\nThe .log file '%s' is needed to figure out when the Wang-Landau weights have been equilibrated, and it was not found.\nYou may rerun with the -x flag and the data will be discarded to 'equiltime', not bothering\nwith the extraction of the information on when the WL weights equilibration was reached.\nOtherwise, put the proper log file into the directory which is subject to the analysis." % logfilename)
            try:
               with open(logfilename, 'r') as infile:
                  dt = float(grepPy(infile, s='delta-t').split()[-1])
                  WLstep = int(grepPy(infile, s='equilibrated').split()[1].replace(':', ''))
            except IndexError:
               parser.error("\nThe Wang-Landau weights haven't equilibrated yet.\nIf you comprehend the consequences, rerun with the -x flag and the data will be discarded to 'equiltime'.")
            WLtime = WLstep * dt
         else:
            WLtime = -1
         return max(WLtime, P.equiltime)

   #===================================================================================================
   # Preliminaries I: Sort the dhdl.xvg files; read in the @-header.
   #===================================================================================================
   
   datafile_tuple = P.datafile_directory, P.prefix, P.suffix
   fs = [ F(filename) for filename in glob( '%s/%s*%s' % datafile_tuple ) ]
   n_files = len(fs)
   
   if not n_files:
      parser.error("\nNo files found within directory '%s' with prefix '%s' and suffix '%s': check your inputs." % datafile_tuple)
   if n_files > 1:
      fs = sorted(fs, key=F.sortedHelper)
   
   if P.bSkipLambdaIndex:
      try:
         lambdas_to_skip = [int(l) for l in trPy(P.bSkipLambdaIndex, '-').split()]
      except:
         parser.error('\n\nDo not understand the format of the string that follows -k.\nIt should be a string of lambda indices linked by "-".')
      fs = [f for f in fs if not f.state in lambdas_to_skip]
      n_files = len(fs)
   
   lv = []  # *** 
   P.snap_size = []
   for nf, f in enumerate(fs):
   
      lv.append(f.readHeader())
   
      if nf>0:
   
         if not f.lv_names == lv_names:
            if not len(f.lv_names) == n_components:
               parser.error("\nFiles do not contain the same number of lambda gradient components; I cannot combine the data.")
            else:
               parser.error("\nThe lambda gradient components have different names; I cannot combine the data.")
         if not f.bPV == bPV:
            parser.error("\nSome files contain the PV energies, some do not; I cannot combine the files.")
   
      else:
   
         P.lv_names = lv_names = f.lv_names
         n_components = len(lv_names)
         bPV = f.bPV
         P.bExpanded = f.bExpanded

   #===================================================================================================
   # Preliminaries II: Analyze data for validity; build up proper 'lv' and count up lambda states 'K'.
   #===================================================================================================
   
   ndE = [len(i) for i in lv]     # ***
   ndE_unique = numpy.unique(ndE) # ***
   
   # Scenario #1: Each file has all the dE columns -- can use MBAR.
   if len(ndE_unique) == 1: # [K]
      if not numpy.array([i == lv[0] for i in lv]).all():
         parser.error("\nArrays of lambda vectors are different; I cannot combine the data.")
      else:
         lv = lv[0]
         # Handle the case when only some particular files/lambdas are given.
         if 1 < n_files < len(lv):
            bSelective_MBAR = True
            sel_states = [f.state for f in fs]
            print sel_states
            lv = [lv[i] for i in sel_states]
         else:
            bSelective_MBAR = False
   
   elif len(ndE_unique) <= 3:
      # Scenario #2: Have the adjacent states only; 2 dE columns for the terminal states, 3 for inner ones.
      if ndE_unique.tolist() == [2, 3]:
         lv  = [l[i>0]  for i,l in enumerate(lv)]
      # Scenario #3: Have a mixture of formats (adjacent and all): either [2,3,K], or [2,K], or [3,K].
      else:
         lv = lv[ndE_unique.argmax()]
      if 'MBAR' in methods:
         print "\nNumber of states is NOT the same for all simulations; I'm assuming that we only evaluate"
         print "nearest neighbor states, and so cannot use MBAR, removing the method."
         methods.remove('MBAR')
      print "\nStitching together the dhdl files. I am assuming that the files are numbered in order of"
      print "increasing lambda; otherwise, results will not be correct."
   
   else:
      print "The files contain the number of the dE columns I cannot deal with; will terminate.\n\n%-10s %s " % ("# of dE's", "File")
      for nf, f in enumerate(fs):
         print "%6d     %s" % (ndE[nf], f.filename)
      parser.error("\nThere are more than 3 groups of files (%s, to be exact) each having different number of the dE columns; I cannot combine the data." % len(ndE_unique))
   
   lv = numpy.array(lv, float) # *** Lambda vectors.
   K  = len(lv)                # *** Number of lambda states.

   #===================================================================================================
   # Preliminaries III: Count up the equilibrated snapshots.
   #===================================================================================================
   
   equiltime = P.equiltime
   nsnapshots = numpy.zeros(K, int)
   
   for nf, f in enumerate(fs):
   
      f.len_first, f.len_last = (len(line.split()) for line in tailPy(f.filename, 2))
      bLenConsistency = (f.len_first != f.len_last)
         
      if f.bExpanded:
   
         equiltime       = f.parseLog()
         equilsnapshots  = int(round(equiltime/f.snap_size))
         f.skip_lines   += equilsnapshots
   
         extract_states  = numpy.genfromtxt(f.filename, dtype=int, skiprows=f.skip_lines, skip_footer=1*bLenConsistency, usecols=1)
         nsnapshots     += numpy.array(Counter(extract_states).values())
   
      else:
         equilsnapshots  = int(equiltime/f.snap_size)
         f.skip_lines   += equilsnapshots
         nsnapshots[nf] += wcPy(f.filename) - f.skip_lines - 1*bLenConsistency
   
      print "First %s ps (%s snapshots) will be discarded due to equilibration from file %s..." % (equiltime, equilsnapshots, f.filename)
   
   #===================================================================================================
   # Preliminaries IV: Load in equilibrated data.
   #===================================================================================================   
   
   maxn  = max(nsnapshots)                                   # maximum number of the equilibrated snapshots from any state
   dhdlt = numpy.zeros([K,n_components,int(maxn)], float)    # dhdlt[k,n,t] is the derivative of energy component n with respect to state k of snapshot t
   u_klt = numpy.zeros([K,K,int(maxn)], numpy.float64)       # u_klt[k,m,t] is the reduced potential energy of snapshot t of state k evaluated at state m
   
   for nf, f in enumerate(fs):
      f.iter_loadtxt(nf)

   return nsnapshots, lv, dhdlt, u_klt

#===================================================================================================
# FUNCTIONS: This is the Sire lambda gradient file parser.
#===================================================================================================

def readDataSire():
   """Read in Sire's output files; return nsnapshots, lv, dhdlt, and u_klt."""

   class F:
      """This is the object to be built on the filename."""

      def __init__(self, filename):
         self.filename   = filename
         self.skip_lines = 0  # Number of lines from the top that are to be skipped.
         snap_size       = [] # Time from first two snapshots to determine snapshot's size.

         print "Reading metadata from %-*s" % (len_fstring+1, self.filename+';'),
         with open(self.filename,'r') as infile:
            for line in infile:

               if line.startswith('#'):
                  self.skip_lines += 1
                  elements = line.split()
                  if 'lambba_val.val' in elements:
                     self.lv = elements[-1]
                     lv.append(elements[-1:])
               else:
                  snap_size.append(float(line.split()[0]))
                  if len(snap_size) > 1:
                     self.snap_size = numpy.diff(snap_size)[0]
                     break
            equilsnapshots  = int(P.equiltime/self.snap_size)
            self.skip_lines += equilsnapshots
            nsnapshots.append(wcPy(infile) + 2 - equilsnapshots)
            print "first %s ps (%s snapshots) will be discarded due to equilibration..." % (P.equiltime, equilsnapshots)

      def loadtxtSire(self, state):
         print "Loading in data from %-*s (state %d) ..." % (len_fstring, self.filename, state)
         dhdlt[state, :, :nsnapshots[state]] = numpy.genfromtxt(self.filename, dtype=float, skiprows=self.skip_lines, usecols=1)
         return

   # Preliminaries I-III: Sort the dhdl.xvg files; read in the header; count up the equilibrated snapshots.
   datafile_tuple = P.datafile_directory, P.prefix, P.suffix
   fs = glob('%s/%s*%s' % datafile_tuple)
   len_fstring = max([len(i) for i in fs])
   K = len(fs)
   if not K:
      parser.error("\nNo files found within directory '%s' with prefix '%s' and suffix '%s': check your inputs." % datafile_tuple)

   n_components = 1
   lv           = []
   nsnapshots   = []
   fs = [ F(filename) for filename in fs ]
   print "\n"
   fs = sorted(fs, key=lambda f: f.lv)
   lv, nsnapshots = zip(*sorted(zip(lv, nsnapshots)))
   lv = numpy.array(lv, float)

   # Preliminaries IV: Load in equilibrated data.
   maxn  = max(nsnapshots)                                   # maximum number of the equilibrated snapshots from any state
   dhdlt = numpy.zeros([K,n_components,int(maxn)], float)    # dhdlt[k,n,t] is the derivative of energy component n with respect to state k of snapshot t
   u_klt = None

   for nf, f in enumerate(fs):
      f.loadtxtSire(nf)

   return nsnapshots, lv, dhdlt, u_klt

#===================================================================================================
# FUNCTIONS: Miscellanea.
#===================================================================================================

def checkUnitsAndMore(units):
   if units == 'kJ':
      beta_report = beta
      units = '(kJ/mol)'
   elif units == 'kcal':
      beta_report = 4.184*beta
      units = '(kcal/mol)'
   elif units == 'kBT':
      beta_report = 1
      units = '(k_BT)'
   else:
      parser.error('\nI don\'t understand the unit type \'%s\': the only options \'kJ\', \'kcal\', and \'kBT\'' % units)

   if not P.output_directory:
      P.output_directory = P.datafile_directory
   if P.overlap:
      if not 'MBAR' in methods:
         parser.error("\nMBAR is not in 'methods'; can't plot the overlap matrix.")

   return units, beta_report

def timeStatistics(stime):
   etime = ttt_time.time()
   tm = int((etime-stime)/60.)
   th = int(tm/60.)
   ts = '%.2f' % (etime-stime-60*(tm+60*th)) 
   return th, tm, ts, ttt_time.asctime()

#===================================================================================================
# FUNCTIONS: The autocorrelation analysis.
#===================================================================================================   

def uncorrelate(sta, fin, do_dhdl=False):
   """Identifies uncorrelated samples and updates the arrays of the reduced potential energy and dhdlt retaining data entries of these samples only.
      'sta' and 'fin' are the starting and final snapshot positions to be read, both are arrays of dimension K."""
   u_kln = numpy.zeros([K,K,max(fin-sta)], numpy.float64) # u_kln[k,m,n] is the reduced potential energy of uncorrelated sample index n from state k evaluated at state m
   N_k = numpy.zeros(K, int) # N_k[k] is the number of uncorrelated samples from state k
   g = numpy.zeros(K,float) # autocorrelation times for the data
   if do_dhdl:
      dhdl = numpy.zeros([K,n_components,max(fin-sta)], float) #dhdl is value for dhdl for each component in the file at each time.
      print "\n\nNumber of correlated and uncorrelated samples:\n\n%6s %12s %12s %12s\n" % ('State', 'N', 'N_k', 'N/N_k')
   for k in range(K):
      # Sum up over the energy components; notice, that only the relevant data is being used in the third dimension.
      dhdl_sum = numpy.sum(dhdlt[k,:,sta[k]:fin[k]], axis=0)
      # Determine indices of uncorrelated samples from potential autocorrelation analysis at state k
      # (alternatively, could use the energy differences -- here, we will use total dhdl).
      g[k] = timeseries.statisticalInefficiency(dhdl_sum)
      indices = numpy.array(timeseries.subsampleCorrelatedData(dhdl_sum, g=g[k])) # indices of uncorrelated samples
      N = len(indices) # number of uncorrelated samples
      # Handle case where we end up with too few.
      if N < P.uncorr_threshold:
         if do_dhdl:
            print "WARNING: Only %s uncorrelated samples found at lambda number %s; proceeding with analysis using correlated samples..." % (N, k)
         indices = numpy.arange(len(dhdl_sum))
         N = len(indices)
      N_k[k] = N # Store the number of uncorrelated samples from state k.
      if not u_klt==None:
         for l in range(K):
            u_kln[k,l,0:N] = u_klt[k,l,indices]
      if do_dhdl:
         print "%6s %12s %12s %12.2f" % (k, fin[k], N_k[k], g[k])
         for n in range(n_components):
            dhdl[k,n,0:N] = dhdlt[k,n,indices]
   if do_dhdl:
      return (dhdl, N_k, u_kln)
   return (N_k, u_kln)

#===================================================================================================
# FUNCTIONS: The MBAR workhorse.
#===================================================================================================   

def estimatewithMBAR(u_kln, N_k, reltol=relative_tolerance, regular_estimate=False):
   """Computes the MBAR free energy given the reduced potential and the number of relevant entries in it."""

   def plotOverlapMatrix(O):
      """Plots the probability of observing a sample from state i (row) in state j (column).
      For convenience, the neigboring state cells are fringed in bold."""
      max_prob = O.max()
      fig = pl.figure(figsize=(K/2.,K/2.))
      fig.add_subplot(111, frameon=False, xticks=[], yticks=[])
   
      for i in range(K):
         if i!=0:
            pl.axvline(x=i, ls='-', lw=0.5, color='k', alpha=0.25)
            pl.axhline(y=i, ls='-', lw=0.5, color='k', alpha=0.25)
         for j in range(K):
            if O[j,i] < 0.005:
               ii = ''
            else:
               ii = ("%.2f" % O[j,i])[1:]
            alf = O[j,i]/max_prob
            pl.fill_between([i,i+1], [K-j,K-j], [K-(j+1),K-(j+1)], color='k', alpha=alf)
            pl.annotate(ii, xy=(i,j), xytext=(i+0.5,K-(j+0.5)), size=8, textcoords='data', va='center', ha='center', color=('k' if alf < 0.5 else 'w'))
   
      cx = sorted(2*range(K+1))
      cy = sorted(2*range(K+1), reverse=True)
      pl.plot(cx[2:-1], cy[1:-2], 'k-', lw=2.0)
      pl.plot(numpy.array(cx[2:-3])+1, cy[1:-4], 'k-', lw=2.0)
      pl.plot(cx[1:-2], numpy.array(cy[:-3])-1, 'k-', lw=2.0)
      pl.plot(cx[1:-4], numpy.array(cy[:-5])-2, 'k-', lw=2.0)
   
      pl.xlim(0, K)
      pl.ylim(0, K)
      pl.savefig(os.path.join(P.output_directory, 'O_MBAR.pdf'), bbox_inches='tight', pad_inches=0.0)
      pl.close(fig)
      return

   MBAR = pymbar.MBAR(u_kln, N_k, verbose = verbose, method = 'adaptive', relative_tolerance = reltol, initialize = P.init_with)
   # Get matrix of dimensionless free energy differences and uncertainty estimate.
   (Deltaf_ij, dDeltaf_ij) = MBAR.getFreeEnergyDifferences(uncertainty_method='svd-ew')
   if verbose: 
      print "Matrix of free energy differences\nDeltaf_ij:\n%s\ndDeltaf_ij:\n%s" % (Deltaf_ij, dDeltaf_ij)
   if regular_estimate:
      if P.overlap: 
         print "The overlap matrix is..."
         O = MBAR.computeOverlap('matrix')
         for k in range(K):
            line = ''
            for l in range(K):
               line += ' %5.2f ' % O[k, l]
            print line
         plotOverlapMatrix(O)
         print "\nFor a nicer figure look at 'O_MBAR.pdf'"
      return (Deltaf_ij, dDeltaf_ij)
   return (Deltaf_ij[0,K-1]/beta_report, dDeltaf_ij[0,K-1]/beta_report)

#===================================================================================================
# FUNCTIONS: Thermodynamic integration.
#===================================================================================================   

class naturalcubicspline:

   def __init__(self, x):

      # define some space
      L = len(x)
      H = numpy.zeros([L,L],float)
      M = numpy.zeros([L,L],float)
      BW = numpy.zeros([L,L],float)
      AW = numpy.zeros([L,L],float)
      DW = numpy.zeros([L,L],float)

      h = x[1:L]-x[0:L-1]
      ih = 1.0/h

      # define the H and M matrix, from p. 371 "applied numerical methods with matlab, Chapra"
      H[0,0] = 1
      H[L-1,L-1] = 1
      for i in range(1,L-1):
         H[i,i] = 2*(h[i-1]+h[i])
         H[i,i-1] = h[i-1]
         H[i,i+1] = h[i]

         M[i,i] = -3*(ih[i-1]+ih[i])
         M[i,i-1] = 3*(ih[i-1])
         M[i,i+1] = 3*(ih[i])

      CW = numpy.dot(numpy.linalg.inv(H),M)  # this is the matrix translating c to weights in f.
                                                   # each row corresponds to the weights for each c.

      # from CW, define the other coefficient matrices
      for i in range(0,L-1):
         BW[i,:]    = -(h[i]/3)*(2*CW[i,:]+CW[i+1,:])
         BW[i,i]   += -ih[i]
         BW[i,i+1] += ih[i]
         DW[i,:]    = (ih[i]/3)*(CW[i+1,:]-CW[i,:])
         AW[i,i]    = 1

      # Make copies of the arrays we'll be using in the future.
      self.x  = x.copy()
      self.AW = AW.copy()
      self.BW = BW.copy()
      self.CW = CW.copy()
      self.DW = DW.copy()

      # find the integrating weights
      self.wsum = numpy.zeros([L],float)
      self.wk = numpy.zeros([L-1,L],float)
      for k in range(0,L-1):
         w = DW[k,:]*(h[k]**4)/4.0 + CW[k,:]*(h[k]**3)/3.0 + BW[k,:]*(h[k]**2)/2.0 + AW[k,:]*(h[k])
         self.wk[k,:] = w
         self.wsum += w

   def interpolate(self,y,xnew):
      if len(self.x) != len(y):
         parser.error("\nThe length of 'y' should be consistent with that of 'self.x'. I cannot perform linear algebra operations.")
      # get the array of actual coefficients by multiplying the coefficient matrix by the values  
      a = numpy.dot(self.AW,y)
      b = numpy.dot(self.BW,y)
      c = numpy.dot(self.CW,y)
      d = numpy.dot(self.DW,y)

      N = len(xnew)
      ynew = numpy.zeros([N],float)
      for i in range(N):
         # Find the index of 'xnew[i]' it would have in 'self.x'.
         j = numpy.searchsorted(self.x, xnew[i]) - 1
         lamw = xnew[i] - self.x[j]
         ynew[i] = d[j]*lamw**3 + c[j]*lamw**2 + b[j]*lamw + a[j]
      # Preserve the terminal points.
      ynew[0] = y[0]
      ynew[-1] = y[-1]
      return ynew

def TIprelim(dhdl, lv):

   # Compute <dhdl> and std(dhdl) for each component, for each lambda; multiply them by beta to make unitless.
   ave_dhdl = numpy.zeros([K,n_components],float)
   std_dhdl = numpy.zeros([K,n_components],float)
   for k in range(K):
      ave_dhdl[k,:] = beta*numpy.average(dhdl[k,:,0:N_k[k]],axis=1)
      std_dhdl[k,:] = beta*numpy.std(dhdl[k,:,0:N_k[k]],axis=1)/numpy.sqrt(N_k[k]-1)

   # Lambda vectors spacing.
   dlam = numpy.diff(lv, axis=0)

   lchange = numpy.zeros([K,n_components],bool)   # booleans for which lambdas are changing 
   for j in range(n_components):
      # need to identify range over which lambda doesn't change, and not interpolate over that range.
      for k in range(K-1):
         if (lv[k+1,j]-lv[k,j] > 0):
            lchange[k,j] = True
            lchange[k+1,j] = True
   return lchange, dlam, ave_dhdl, std_dhdl

def getSplines(lchange):
   # construct a map back to the original components
   mapl = numpy.zeros([K,n_components],int)   # map back to the original k from the components
   for j in range(n_components):
      incr = 0
      for k in range(K):
         if (lchange[k,j]):
            mapl[k,j] += incr
            incr +=1

   # put together the spline weights for the different components
   cubspl = list()
   for j in range(n_components):
      lv_lchange = lv[lchange[:,j],j]
      if len(lv_lchange) == 0: # handle the all-zero lv column
         cubspl.append(0)
      else:
         spl = naturalcubicspline(lv_lchange)
         cubspl.append(spl)
   return cubspl, mapl

#===================================================================================================
# FUNCTIONS: This one estimates dF and ddF for all pairs of adjacent states and stores them.
#===================================================================================================   

def estimatePairs():

   print ("Estimating the free energy change with %s..." % ', '.join(methods)).replace(', MBAR', '')
   df_allk = list(); ddf_allk = list()
   
   for k in range(K-1):
      df = dict(); ddf = dict()
   
      for name in methods:
   
         if name == 'TI':
            #===================================================================================================
            # Estimate free energy difference with TI; interpolating with the trapezoidal rule.
            #===================================================================================================   
            df['TI'] = 0.5*numpy.dot(dlam[k],(ave_dhdl[k]+ave_dhdl[k+1]))        
            ddf['TI'] = 0.5*numpy.sqrt(numpy.dot(dlam[k]**2,std_dhdl[k]**2+std_dhdl[k+1]**2))               
   
         if name == 'TI-CUBIC':
            #===================================================================================================
            # Estimate free energy difference with TI; interpolating with the natural cubic splines.
            #===================================================================================================   
            df['TI-CUBIC'], ddf['TI-CUBIC'] = 0, 0
            for j in range(n_components):
               if dlam[k,j] > 0:
                  lj = lchange[:,j]
                  df['TI-CUBIC'] += numpy.dot(cubspl[j].wk[mapl[k,j]],ave_dhdl[lj,j])
                  ddf['TI-CUBIC'] += numpy.dot(cubspl[j].wk[mapl[k,j]]**2,std_dhdl[lj,j]**2)
            ddf['TI-CUBIC'] = numpy.sqrt(ddf['TI-CUBIC'])
   
         if any(name == m for m in ['DEXP', 'GDEL', 'BAR', 'UBAR', 'RBAR']):
            w_F = u_kln[k,k+1,0:N_k[k]] - u_kln[k,k,0:N_k[k]] 
   
         if name == 'DEXP':
            #===================================================================================================
            # Estimate free energy difference with Forward-direction EXP (in this case, Deletion from solvent).
            #===================================================================================================   
            (df['DEXP'], ddf['DEXP']) = pymbar.computeEXP(w_F)
   
         if name == 'GDEL':
            #===================================================================================================
            # Estimate free energy difference with a Gaussian estimate of EXP (in this case, deletion from solvent)
            #===================================================================================================   
            (df['GDEL'], ddf['GDEL']) = pymbar.computeGauss(w_F)
   
         if any(name == m for m in ['IEXP', 'GINS', 'BAR', 'UBAR', 'RBAR']):
            w_R = u_kln[k+1,k,0:N_k[k+1]] - u_kln[k+1,k+1,0:N_k[k+1]] 
   
         if name == 'IEXP':
            #===================================================================================================
            # Estimate free energy difference with Reverse-direction EXP (in this case, insertion into solvent).
            #===================================================================================================   
            (rdf,rddf) = pymbar.computeEXP(w_R)
            (df['IEXP'], ddf['IEXP']) = (-rdf,rddf)
   
         if name == 'GINS':
            #===================================================================================================
            # Estimate free energy difference with a Gaussian estimate of EXP (in this case, insertion into solvent)
            #===================================================================================================   
            (rdf,rddf) = pymbar.computeGauss(w_R)
            (df['GINS'], ddf['GINS']) = (-rdf,rddf)
   
         if name == 'BAR':
            #===================================================================================================
            # Estimate free energy difference with BAR; use w_F and w_R computed above.
            #===================================================================================================   
            (df['BAR'], ddf['BAR']) = pymbar.computeBAR(w_F, w_R, relative_tolerance=relative_tolerance, verbose = verbose)      
   
         if name == 'UBAR':
            #===================================================================================================
            # Estimate free energy difference with unoptimized BAR -- assume dF is zero, and just do one evaluation
            #===================================================================================================   
            (df['UBAR'], ddf['UBAR']) = pymbar.computeBAR(w_F, w_R, verbose = verbose,iterated_solution=False)
   
         if name == 'RBAR':
            #===================================================================================================
            # Estimate free energy difference with Unoptimized BAR over range of free energy values, and choose the one 
            # that is self consistently best.
            #===================================================================================================   
            min_diff = 1E6
            best_udf = 0
            for trial_udf in range(-10,10,1):
               (udf, uddf) = pymbar.computeBAR(w_F, w_R, DeltaF=trial_udf, iterated_solution=False, verbose=verbose)
               diff = numpy.abs(udf - trial_udf)
               if (diff < min_diff):
                  best_udf = udf
                  best_uddf = uddf
                  min_diff = diff
            (df['RBAR'], ddf['RBAR']) = (best_udf,best_uddf)
   
         if name == 'MBAR':
            #===================================================================================================
            # Store the MBAR free energy difference (already estimated above) properly, i.e. by state.
            #===================================================================================================   
            (df['MBAR'], ddf['MBAR']) =  Deltaf_ij[k,k+1], dDeltaf_ij[k,k+1]
   
      df_allk = numpy.append(df_allk,df)
      ddf_allk = numpy.append(ddf_allk,ddf)

   return df_allk, ddf_allk

#===================================================================================================
# FUNCTIONS: All done with calculations; summarize and print stats.
#===================================================================================================   

def totalEnergies():

   # Count up the charging states.
   numcharging = 0
   for lv_n in ['coul', 'fep']:
      if lv_n in P.lv_names:
         ndx_char = P.lv_names.index(lv_n)
         lv_char = lv[:, ndx_char]
         if not (lv_char == lv_char[0]).all():
            numcharging = (lv_char != 1).sum()
            break
   if numcharging == K:
      numcharging = K-1
   
   # Split the total energies into segments; initialize lists to store them.
   segments      = ['Coulomb'  , 'vdWaals'  , 'TOTAL']
   segmentstarts = [0          , numcharging, 0      ]
   segmentends   = [numcharging, K-1        , K-1    ]
   dFs  = []
   ddFs = []
   
   # Perform the energy segmentation; be pedantic about the TI cumulative ddF's (see Section 3.1 of the paper).
   for i in range(len(segments)):
      segment = segments[i]; segstart = segmentstarts[i]; segend = segmentends[i]
      dF  = dict.fromkeys(methods, 0)
      ddF = dict.fromkeys(methods, 0)
   
      for name in methods:
         if name == 'MBAR':
            dF['MBAR']  =  Deltaf_ij[segstart, segend]
            ddF['MBAR'] = dDeltaf_ij[segstart, segend]
   
         elif name[0:2] == 'TI':
            for k in range(segstart, segend):
               dF[name] += df_allk[k][name]
   
            if segment == 'Coulomb':
               jlist = [ndx_char] if numcharging>0 else []
            elif segment == 'vdWaals':
               jlist = []
            elif segment == 'TOTAL':
               jlist = range(n_components)
   
            for j in jlist:
               lj = lchange[:,j]
               if not (lj == False).all(): # handle the all-zero lv column
                  if name == 'TI-CUBIC':
                     ddF[name] += numpy.dot((cubspl[j].wsum)**2,std_dhdl[lj,j]**2)
                  elif name == 'TI':
                     h = numpy.trim_zeros(dlam[:,j])
                     wsum = 0.5*(numpy.append(h,0) + numpy.append(0,h))
                     ddF[name] += numpy.dot(wsum**2,std_dhdl[lj,j]**2)
            ddF[name] = numpy.sqrt(ddF[name])
   
         else:
            for k in range(segstart,segend):
               dF[name] += df_allk[k][name]
               ddF[name] += (ddf_allk[k][name])**2
            ddF[name] = numpy.sqrt(ddF[name])
   
      dFs.append(dF)
      ddFs.append(ddF)
   
   for name in methods: # 'vdWaals' = 'TOTAL' - 'Coulomb'
      ddFs[1][name] = (ddFs[2][name]**2 - ddFs[0][name]**2)**0.5
   
   # Display results.
   def printLine(str1, str2, d1=None, d2=None):
      """Fills out the results table linewise."""
      print str1,
      text = str1
      for name in methods:
         if d1 == 'plain':
            print str2,
            text += ' ' + str2
         if d1 == 'name':
            print str2 % (name, units),
            text += ' ' + str2 % (name, units)
         if d1 and d2:
            print str2 % (d1[name]/beta_report, d2[name]/beta_report),
            text += ' ' + str2 % (d1[name]/beta_report, d2[name]/beta_report)
      print ''
      outtext.append(text + '\n')
      return
   
   outtext = []
   printLine(12*'-', 21*'-', 'plain')
   printLine('%-12s' % '   States', '%9s %-11s', 'name')
   printLine(12*'-', 21*'-', 'plain')
   for k in range(K-1):
      printLine('%4d -- %-4d' % (k, k+1), '%10.3f  +- %6.3f', df_allk[k], ddf_allk[k])
   printLine(12*'-', 21*'-', 'plain')
   for i in range(len(segments)):
      printLine('%9s:  ' % segments[i], '%10.3f  +- %6.3f', dFs[i], ddFs[i])
   print """\n\n
                                 In the table shown are the free energy differences computed by means of various methods
                                 for each pair of adjacent states with 'TOTAL' being the sum of the rows, thus yielding
                                 the free energy difference between the terminal states 0 and %d.\n\n
                                 A remark on the energy components interpretation:
                                 'vdWaals' is computed as ('TOTAL' - 'Coulomb'), where 'Coulomb' is thought of as the free
                                 energy change between the states defined by the lambda vectors (0,0,...,0) and (1,0,...,0),
                                 the only varying vector component being either 'coul-lambda' or 'fep-lambda'.""" % (K-1)
   # Store results.
   file = open(os.path.join(P.output_directory, 'results.txt'), 'w')
   file.writelines(outtext)
   file.close()

   P.datafile_directory = os.getcwd()
   P.when_analyzed = ttt_time.asctime()
   P.ddf_allk = ddf_allk
   P.df_allk  = df_allk
   P.ddFs     = ddFs
   P.dFs      = dFs

   file = open(os.path.join(P.output_directory, 'results.pickle'), 'w')
   pickle.dump(P, file)
   file.close()

   return

#===================================================================================================
# FUNCTIONS: Free energy change vs. simulation time. Called by the -f flag.
#===================================================================================================   

def dF_t():

   def plotdFvsTime(f_ts, r_ts, F_df, R_df, F_ddf, R_ddf):
      """Plots the free energy change computed using the equilibrated snapshots between the proper target time frames (f_ts and r_ts)
      in both forward (data points are stored in F_df and F_ddf) and reverse (data points are stored in R_df and R_ddf) directions."""
      fig = pl.figure(figsize=(8,6))
      ax = fig.add_subplot(111)
      pl.setp(ax.spines['bottom'], color='#D2B9D3', lw=3, zorder=-2)
      pl.setp(ax.spines['left'], color='#D2B9D3', lw=3, zorder=-2)
      for dire in ['top', 'right']:
         ax.spines[dire].set_color('none')
      ax.xaxis.set_ticks_position('bottom')
      ax.yaxis.set_ticks_position('left')
   
      max_fts = max(f_ts)
      rr_ts = [aa/max_fts for aa in f_ts[::-1]]
      f_ts = [aa/max_fts for aa in f_ts]
      r_ts = [aa/max_fts for aa in r_ts]

      line0  = pl.fill_between([r_ts[0], f_ts[-1]], R_df[0]-R_ddf[0], R_df[0]+R_ddf[0], color='#D2B9D3', zorder=-5)
      for i in range(len(f_ts)):
         line1 = pl.plot([f_ts[i]]*2, [F_df[i]-F_ddf[i], F_df[i]+F_ddf[i]], color='#736AFF', ls='-', lw=3, solid_capstyle='round', zorder=1)
      line11 = pl.plot(f_ts, F_df, color='#736AFF', ls='-', lw=3, marker='o', mfc='w', mew=2.5, mec='#736AFF', ms=12, zorder=2)
 
      for i in range(len(rr_ts)):
         line2 = pl.plot([rr_ts[i]]*2, [R_df[i]-R_ddf[i], R_df[i]+R_ddf[i]], color='#C11B17', ls='-', lw=3, solid_capstyle='round', zorder=3)
      line22 = pl.plot(rr_ts, R_df, color='#C11B17', ls='-', lw=3, marker='o', mfc='w', mew=2.5, mec='#C11B17', ms=12, zorder=4)
   
      pl.xlim(r_ts[0], f_ts[-1])
   
      pl.xticks(r_ts[::2] + f_ts[-1:], fontsize=10)
      pl.yticks(fontsize=10)
   
      leg = pl.legend((line1[0], line2[0]), (r'$Forward$', r'$Reverse$'), loc=9, prop=FP(size=18), frameon=False)
      pl.xlabel(r'$\mathrm{Fraction\/of\/the\/simulation\/time}$', fontsize=16, color='#151B54')
      pl.ylabel(r'$\mathrm{\Delta G\/%s}$' % units, fontsize=16, color='#151B54')
      pl.xticks(f_ts, ['%.2f' % i for i in f_ts])
      pl.tick_params(axis='x', color='#D2B9D3')
      pl.tick_params(axis='y', color='#D2B9D3')
      pl.savefig(os.path.join(P.output_directory, 'dF_t.pdf'))
      pl.close(fig)
      return

   if not 'MBAR' in methods:
      parser.error("\nCurrent version of the dF(t) analysis works with MBAR only and the method is not found in the list.")
   if not (P.snap_size[0] == numpy.array(P.snap_size)).all(): # this could be circumvented
      parser.error("\nThe snapshot size isn't the same for all the files; cannot perform the dF(t) analysis.")

   # Define a list of bForwrev equidistant time frames at which the free energy is to be estimated; count up the snapshots embounded between the time frames.
   n_tf = P.bForwrev + 1
   nss_tf = numpy.zeros([n_tf, K], int)
   increment = 1./(n_tf-1)
   if P.bExpanded:
      tf = numpy.arange(0,1+increment,increment)*(numpy.sum(nsnapshots)-1)+1
      tf[0] = 0
      for i in range(n_tf-1):
         nss = Counter(extract_states[tf[i]:tf[i+1]])       
         nss_tf[i+1] = numpy.array([nss[j] for j in range(K)])
   else:
      tf = numpy.arange(0,1+increment,increment)*(max(nsnapshots)-1)+1
      tf[0] = 0
      for i in range(n_tf-1):
         nss_tf[i+1] = numpy.array([min(j, tf[i+1]) for j in nsnapshots]) - numpy.sum(nss_tf[:i+1],axis=0)

   # Define the real time scale (in ps) rather than a snapshot sequence.
   ts = ["%.1f" % ((i-(i!=tf[0]))*P.snap_size[0] + P.equiltime) for i in tf]
   # Initialize arrays to store data points to be plotted.
   F_df  = numpy.zeros(n_tf-1, float)
   F_ddf = numpy.zeros(n_tf-1, float)
   R_df  = numpy.zeros(n_tf-1, float)
   R_ddf = numpy.zeros(n_tf-1, float)
   # Store the MBAR energy that accounts for all the equilibrated snapshots (has already been computed in the previous section).
   F_df[-1], F_ddf[-1] = (Deltaf_ij[0,K-1]/beta_report, dDeltaf_ij[0,K-1]/beta_report)
   R_df[0], R_ddf[0]   = (Deltaf_ij[0,K-1]/beta_report, dDeltaf_ij[0,K-1]/beta_report)
   # Do the forward analysis.
   print "Forward dF(t) analysis...\nEstimating the free energy change using the data up to"
   sta = nss_tf[0]
   for i in range(n_tf-2):
      print "%60s ps..." % ts[i+1]
      fin = numpy.sum(nss_tf[:i+2],axis=0)
      N_k, u_kln = uncorrelate(nss_tf[0], numpy.sum(nss_tf[:i+2],axis=0))
      F_df[i], F_ddf[i] = estimatewithMBAR(u_kln, N_k)
      a, b = estimatewithMBAR(u_kln, N_k)
   # Do the reverse analysis.
   print "Reverse dF(t) analysis...\nUsing the data starting from"
   fin = numpy.sum(nss_tf[:],axis=0)
   for i in range(n_tf-2):
      print "%34s ps..." % ts[i+1]
      sta = numpy.sum(nss_tf[:i+2],axis=0)
      N_k, u_kln = uncorrelate(sta, fin)
      R_df[i+1], R_ddf[i+1] = estimatewithMBAR(u_kln, N_k)

   print """\n   The free energies %s evaluated by using the trajectory
   snaphots corresponding to various time intervals for both the
   reverse and forward (in parentheses) direction.\n""" % units
   print "%s\n %20s %19s %20s\n%s" % (70*'-', 'Time interval, ps','Reverse', 'Forward', 70*'-')
   print "%10s -- %s\n%10s -- %-10s %11.3f +- %5.3f %16s\n" % (ts[0], ts[-1], '('+ts[0], ts[0]+')', R_df[0], R_ddf[0], 'XXXXXX')
   for i in range(1, len(ts)-1):
      print "%10s -- %s\n%10s -- %-10s %11.3f +- %5.3f %11.3f +- %5.3f\n" % (ts[i], ts[-1], '('+ts[0], ts[i]+')', R_df[i], R_ddf[i], F_df[i-1], F_ddf[i-1])
   print "%10s -- %s\n%10s -- %-10s %16s %15.3f +- %5.3f\n%s" % (ts[-1], ts[-1], '('+ts[0], ts[-1]+')', 'XXXXXX', F_df[-1], F_ddf[-1], 70*'-')

   # Plot the forward and reverse dF(t); store the data points in the text file.
   print "Plotting data to the file dF_t.pdf...\n\n"
   plotdFvsTime([float(i) for i in ts[1:]], [float(i) for i in ts[:-1]], F_df, R_df, F_ddf, R_ddf)
   outtext = ["%12s %10s %-10s %17s %10s %s\n" % ('Time (ps)', 'Forward', units, 'Time (ps)', 'Reverse', units)]
   outtext+= ["%10s %11.3f +- %5.3f %18s %11.3f +- %5.3f\n" % (ts[1:][i], F_df[i], F_ddf[i], ts[:-1][i], R_df[i], R_ddf[i]) for i in range(len(F_df))]
   file = open(os.path.join(P.output_directory, 'dF_t.txt'), 'w'); file.writelines(outtext); file.close()
   return

#===================================================================================================
# FUNCTIONS: Free energy change breakdown (into lambda-pair dFs). Called by the -g flag.
#===================================================================================================   

def plotdFvsLambda():

   def plotdFvsLambda1():
      """Plots the free energy differences evaluated for each pair of adjacent states for all methods."""
      x = numpy.arange(len(df_allk))
      if x[-1]<8:
         fig = pl.figure(figsize = (8,6))
      else:
         fig = pl.figure(figsize = (len(x),6))
      width = 1./(len(methods)+1)
      elw = 30*width
      colors = {'TI':'#C45AEC', 'TI-CUBIC':'#33CC33', 'DEXP':'#F87431', 'IEXP':'#FF3030', 'GINS':'#EAC117', 'GDEL':'#347235', 'BAR':'#6698FF', 'UBAR':'#817339', 'RBAR':'#C11B17', 'MBAR':'#F9B7FF'}
      lines = tuple()
      for name in methods:
         y = [df_allk[i][name]/beta_report for i in x]
         ye = [ddf_allk[i][name]/beta_report for i in x]
         line = pl.bar(x+len(lines)*width, y, width, color=colors[name], yerr=ye, lw=0.1*elw, error_kw=dict(elinewidth=elw, ecolor='black', capsize=0.5*elw))
         lines += (line[0],)
      pl.xlabel('States', fontsize=12, color='#151B54')
      pl.ylabel('$\Delta F$ '+units, fontsize=12, color='#151B54')
      pl.xticks(x+0.5*width*len(methods), tuple(['%d--%d' % (i, i+1) for i in x]), fontsize=8)
      pl.yticks(fontsize=8)
      pl.xlim(x[0], x[-1]+len(lines)*width)
      ax = pl.gca()
      for dir in ['right', 'top', 'bottom']:
         ax.spines[dir].set_color('none')
      ax.yaxis.set_ticks_position('left')
      for tick in ax.get_xticklines():
         tick.set_visible(False)
 
      leg = pl.legend(lines, tuple(methods), loc=3, ncol=2, prop=FP(size=10), fancybox=True)
      leg.get_frame().set_alpha(0.5)
      pl.title('The free energy change breakdown', fontsize = 12)
      pl.savefig(os.path.join(P.output_directory, 'dF_state_long.pdf'), bbox_inches='tight')
      pl.close(fig)
      return
 
   def plotdFvsLambda2(nb=10):
      """Plots the free energy differences evaluated for each pair of adjacent states for all methods.
      The layout is approximately 'nb' bars per subplot."""
      x = numpy.arange(len(df_allk))
      if len(x) < nb:
         return
      xs = numpy.array_split(x, len(x)/nb+1)
      mnb = max([len(i) for i in xs])
      fig = pl.figure(figsize = (8,6))
      width = 1./(len(methods)+1)
      elw = 30*width
      colors = {'TI':'#C45AEC', 'TI-CUBIC':'#33CC33', 'DEXP':'#F87431', 'IEXP':'#FF3030', 'GINS':'#EAC117', 'GDEL':'#347235', 'BAR':'#6698FF', 'UBAR':'#817339', 'RBAR':'#C11B17', 'MBAR':'#F9B7FF'}
      ndx = 1
      for x in xs:
         lines = tuple()
         ax = pl.subplot(len(xs), 1, ndx)
         for name in methods:
            y = [df_allk[i][name]/beta_report for i in x]
            ye = [ddf_allk[i][name]/beta_report for i in x]
            line = pl.bar(x+len(lines)*width, y, width, color=colors[name], yerr=ye, lw=0.05*elw, error_kw=dict(elinewidth=elw, ecolor='black', capsize=0.5*elw))
            lines += (line[0],)
         for dir in ['left', 'right', 'top', 'bottom']:
            if dir == 'left':
               ax.yaxis.set_ticks_position(dir)
            else:
               ax.spines[dir].set_color('none')
         pl.yticks(fontsize=10)
         ax.xaxis.set_ticks([])
         for i in x+0.5*width*len(methods):
            ax.annotate('$\mathrm{%d-%d}$' % (i, i+1), xy=(i, 0), xycoords=('data', 'axes fraction'), xytext=(0, -2), size=10, textcoords='offset points', va='top', ha='center')
         pl.xlim(x[0], x[-1]+len(lines)*width + (mnb - len(x)))
         ndx += 1
      leg = ax.legend(lines, tuple(methods), loc=0, ncol=2, prop=FP(size=8), title='$\mathrm{\Delta G\/%s\/}\mathit{vs.}\/\mathrm{lambda\/pair}$' % units, fancybox=True)
      leg.get_frame().set_alpha(0.5)
      pl.savefig(os.path.join(P.output_directory, 'dF_state.pdf'), bbox_inches='tight')
      pl.close(fig)
      return

   def plotTI():
      """Plots the ave_dhdl array as a function of the lambda value.
      If (TI and TI-CUBIC in methods) -- plots the TI integration area and the TI-CUBIC interpolation curve,
      elif (only one of them in methods) -- plots the integration area of the method."""
      min_dl = dlam[dlam != 0].min()
      S = int(0.4/min_dl)
      if S>19918:
         fig = pl.figure(figsize = (S,6))
      else:
         fig = pl.figure(figsize = (8,6))
      ax = fig.add_subplot(1,1,1)
      ax.spines['bottom'].set_position('zero')
      ax.spines['top'].set_color('none')
      ax.spines['right'].set_color('none')
      ax.xaxis.set_ticks_position('bottom')
      ax.yaxis.set_ticks_position('left')

      xs, ndx, dx = [0], 0, 0.001
      colors = ['r', 'g', '#7F38EC', '#9F000F', 'b', 'y']
      min_y, max_y = 0, 0

      lines = tuple()
      ## lv_names2 = [r'$Coulomb$', r'$vdWaals$'] ## for the paper
      lv_names2 = [r'$%s$' % string_i.capitalize() for string_i in P.lv_names]

      for j in range(n_components):
         y = ave_dhdl[:,j]
         if not (y == 0).all():
            #if not cubspl[j] == 0:

            # Get the coordinates.
            lj = lchange[:,j]
            x = lv[:,j][lj]
            y = y[lj]/beta_report

            if 'TI' in methods:
               # Plot the TI integration area.
               ss = 'TI'
               for i in range(len(x)-1):
                  min_y = min(y.min(), min_y)
                  max_y = max(y.max(), max_y)
                  #pl.plot(x,y)
                  if i%2==0:
                     pl.fill_between(x[i:i+2]+ndx, 0, y[i:i+2], color=colors[ndx], alpha=1.0)
                  else:
                     pl.fill_between(x[i:i+2]+ndx, 0, y[i:i+2], color=colors[ndx], alpha=0.5)
               xlegend = [-100*wnum for wnum in range(len(lv_names2))]
               pl.plot(xlegend, [0*wnum for wnum in xlegend], ls='-', color=colors[ndx], label=lv_names2[ndx]) ## for the paper

               if 'TI-CUBIC' in methods:
                  # Plot the TI-CUBIC interpolation curve.
                  ss += ' and TI-CUBIC'
                  xnew = numpy.arange(0, 1+dx, dx)
                  ynew = cubspl[j].interpolate(y, xnew)
                  min_y = min(ynew.min(), min_y)
                  max_y = max(ynew.max(), max_y)
                  pl.plot(xnew+ndx, ynew, color='#B6B6B4', ls ='-', solid_capstyle='round', lw=3.0)

            else:
               # Plot the TI-CUBIC integration area.
               ss = 'TI-CUBIC'
               for i in range(len(x)-1):
                  xnew = numpy.arange(x[i], x[i+1]+dx, dx)
                  ynew = cubspl[j].interpolate(y, xnew)
                  ynew[0], ynew[-1] = y[i], y[i+1]
                  min_y = min(ynew.min(), min_y)
                  max_y = max(ynew.max(), max_y)
                  if i%2==0:
                     pl.fill_between(xnew+ndx, 0, ynew, color=colors[ndx], alpha=1.0)
                  else:
                     pl.fill_between(xnew+ndx, 0, ynew, color=colors[ndx], alpha=0.5)

            # Store the abscissa values and update the subplot index.
            xs += (x+ndx).tolist()[1:]
            ndx += 1

      # Make sure the tick labels are not overcrowded.
      xt = range(K)
      min_dl = 0.0155
      if S>5:
         i = 0
         while i < len(xs)-1:
            if i==10:
               min_dl *= 2
            if xs[i+1]-xs[i] < min_dl:
               xt[i+1] = ''
               i += 1
            i += 1
      pl.xticks(xs[1:], xt[1:], fontsize=10)
      pl.yticks(fontsize=10)

      # Remove the abscissa ticks and set up the axes limits.
      for tick in ax.get_xticklines():
         tick.set_visible(False)
      pl.xlim(0, ndx)
      min_y *= 1.01
      max_y *= 1.01
      pl.ylim(min_y, max_y)

      for i,j in zip(xs[1:], xt[1:]):
         pl.annotate(('%.2f' % (i-1.0 if i>1.0 else i) if not j=='' else ''), xy=(i, 0), xytext=(i, 0.01), size=10, rotation=90, textcoords=('data', 'axes fraction'), va='bottom', ha='center', color='#151B54')
      if ndx>1:
         lenticks = len(ax.get_ymajorticklabels()) - 1
         if min_y<0: lenticks -= 1
         if lenticks < 5:
            from matplotlib.ticker import AutoMinorLocator as AML
            ax.yaxis.set_minor_locator(AML())
      pl.grid(which='both', color='w', lw=0.25, axis='y', zorder=12)
      pl.ylabel(r'$\mathrm{\langle{\frac{ \partial U } { \partial \lambda }}\rangle_{\lambda}\/%s}$' % units, fontsize=20, color='#151B54')
      #pl.xlabel('$\mathit{\lambda}$', fontsize=20, color='#151B54')
      pl.annotate('$\mathit{\lambda}$', xy=(0, 0), xytext=(0.5, -0.05), size=18, textcoords='axes fraction', va='top', ha='center', color='#151B54')
      #lege = ax.legend((l1, l2) ('l1', 'l2'), loc='best')
      if not P.software=='Sire':
         lege = ax.legend(prop=FP(size=14), frameon=False, loc=1)
         for l in lege.legendHandles:
            l.set_linewidth(10)
      pl.savefig(os.path.join(P.output_directory, 'dhdl_TI.pdf'))
      pl.close(fig)
      return

   plotdFvsLambda1()
   plotdFvsLambda2()
   if ('TI' in methods or 'TI-CUBIC' in methods):
      plotTI()

#===================================================================================================
# FUNCTIONS: FIG. 7
#===================================================================================================   

def plotCFM(u_kln, num_bins=50):
   """A graphical representation of what Bennett calls 'Curve-Fitting Method'."""

   def leaveTicksOnlyOnThe(xdir, ydir, axis):
      dirs = ['left', 'right', 'top', 'bottom']
      axis.xaxis.set_ticks_position(xdir)
      axis.yaxis.set_ticks_position(ydir)
      return

   def plotFigure7(yy, df_allk, ddf_allk):
      fig = pl.figure(figsize = (8,6))
      matplotlib.rc('axes', facecolor = '#E3E4FA')
      matplotlib.rc('axes', edgecolor = 'white')
      sq = (len(yy))**0.5
      h = int(sq)
      w = h + 1 + 1*(sq-h>0.5)
      for i, (xx_i, yy_i) in enumerate(yy):
         ax = pl.subplot(h, w, i+1)
         ax.plot(xx_i, yy_i, color='r', ls='-', lw=3, marker='o', mec='r')
         leaveTicksOnlyOnThe('bottom', 'left', ax)
         ax.fill_between(xx_i, df_allk[i]['BAR'] - ddf_allk[i]['BAR'], df_allk[i]['BAR'] + ddf_allk[i]['BAR'], color='#D2B9D3', zorder=-1)
         ax.annotate(r'$\mathrm{%d-%d}$' % (i, i+1), xy=(0.5, 0.9), xycoords=('axes fraction', 'axes fraction'), xytext=(0, -2), size=14, textcoords='offset points', va='top', ha='center', color='#151B54', bbox = dict(fc='w', ec='none', boxstyle='round', alpha=0.5))
         pl.xlim(xx_i.min(), xx_i.max())
      #pl.subplots_adjust(left=0.1, bottom=0.1, wspace=0.1)
      pl.suptitle(r'$\mathrm{\Delta g_{i+1,i}\/\ vs\/\/\Delta U_{i,i+1}\/(reduced\/units)}$', fontsize=20, color='#151B54')
      pl.savefig('Fig.7.pdf')
      pl.close(fig)
      return

   K = len(u_kln)
   yy = []
   xx = []
   for k in range(0, K-1):
      righ = -u_kln[k,k+1]
      left = u_kln[k+1,k]
      min1 = min(left.min(), righ.min())
      max2 = max(righ.max(), left.max())
      (counts_l, xbins_l) = numpy.histogram(left, bins=num_bins, range=(min1, max2))
      (counts_r, xbins_r) = numpy.histogram(righ, bins=num_bins, range=(min1, max2))

      log_left = numpy.log(counts_l) - 0.5*xbins_l[:-1]
      log_righ = numpy.log(counts_r) + 0.5*xbins_r[:-1]
      diff = log_left - log_righ
      yy.append((xbins_l[:-1], diff))

   plotFigure7(yy, df_allk, ddf_allk)
   return

#===================================================================================================
# MAIN
#===================================================================================================

if __name__ == "__main__":

   # Timing.
   stime = ttt_time.time()
   print "Started on %s" % ttt_time.asctime()

   # Simulation profile P (to be stored in 'results.pickle') will amass information about the simulation.
   P       = parser.parse_args()[0]
   verbose = P.verbose
   beta    = 1./(kB*P.temperature) if not P.software=='Sire' else 4.184/(kB*P.temperature) # TODO: change the output units in Sire

   units, beta_report = checkUnitsAndMore(P.units)

   if P.software == 'Gromacs':
      nsnapshots, lv, dhdlt, u_klt = readDataGromacs()
   elif P.software == 'Sire':
      nsnapshots, lv, dhdlt, u_klt = readDataSire()
      methods = [i for i in methods if i[:2]=='TI']
      P.lv_names = ['']
   else:
      from inspect import currentframe, getframeinfo
      lineno = getframeinfo(currentframe()).lineno
      print "\n\n%s\n You are analyzing data files that come from neither Gromacs nor Sire. \n Please modify lines %d and %d of this script.\n%s\n\n" % (78*"*", lineno+3, lineno+4, 78*"*")
      #### LINES TO BE MODIFIED
      import YOUR_OWN_MODULE
      nsnapshots, lv, dhdlt, u_klt = YOUR_OWN_MODULE.yourDataParser(*args, **kwargs)
      #### All the four are numpy arrays.
      #### lv           is the array of lambda vectors.
      #### nsnapshots   is the number of equilibrated snapshots per each state.
      #### dhdlt[k,n,t] is the derivative of energy component n with respect to state k of snapshot t
      #### u_klt[k,m,t] is the reduced potential energy of snapshot t of state k evaluated at state m

   K, n_components = lv.shape
   dhdl, N_k, u_kln = uncorrelate(sta=numpy.zeros(K, int), fin=nsnapshots, do_dhdl=True)

   # Estimate free energy difference with MBAR -- all states at once.
   if 'MBAR' in methods:
      print "\nEstimating the free energy change with MBAR..."
      Deltaf_ij, dDeltaf_ij = estimatewithMBAR(u_kln, N_k, regular_estimate=True)

   # The TI preliminaries.
   if ('TI' in methods or 'TI-CUBIC' in methods):
      lchange, dlam, ave_dhdl, std_dhdl = TIprelim(dhdl, lv)
   if 'TI-CUBIC' in methods:
      cubspl, mapl = getSplines(lchange)

   # Call other methods. Print stats. Store results.
   df_allk, ddf_allk = estimatePairs()
   totalEnergies()

   # Plot figures.
   if P.bForwrev:
      dF_t()
   if P.breakdown:
      plotdFvsLambda()
      if not u_kln==None:
         plotCFM(u_kln)

   print "\n\nTime spent: %s hours, %s minutes, and %s seconds.\nFinished on %s" % timeStatistics(stime)
#===================================================================================================
#                                   End of the script 
#===================================================================================================
