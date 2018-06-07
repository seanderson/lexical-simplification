    
def write_index_file(OUTINDEX,filenames,numpars,sindexlst):
    '''Writes file of articles/pars/sentences in format:
    #articles
    artname  numpars numS_par0 numS_par1 ...
    '''
    sidx = 0
    with open(OUTINDEX,'w') as findex:
        findex.write("%d\n" % len(filenames))
        for i,f in enumerate(filenames):
            findex.write("%s %d" % (f,numpars[i]))
            for j in range(numpars[i]):
                findex.write(" %d" % sindexlst[sidx])
                sidx += 1
            findex.write("\n")

def read_index_file(INDEXFILE):
    '''Reads index file.'''
    fnames = [ ]
    numpars = [ ]
    sindexlst = [ ]
    with open(INDEXFILE,'r') as findex:
        n = int(findex.readline().rstrip('\n'))

        for i in range(n):
            dat = findex.readline().rstrip('\n').split()
            fnames.append( dat[0] )
            numpars.append(int(dat[1]))
            sindexlst.append( map(int,dat[2:]))
    return fnames,numpars,sindexlst

