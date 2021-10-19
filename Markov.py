import random
import pandas as pd
import math
import numpy as np

class mini_markov :
    """ 
    Class mini_markov for simple Markov Chain Monte Carlo state based simulations
    
    """
    def __init__(self):

        self.log = None
        self.logtype = 'none'
        self._outcols = ['Trial', 'Epoch', 'State']
        self._outrow = 0
        self.outdf = pd.DataFrame(columns=self._outcols)
        
    def __version__(self):
        return '0.0.1'

    def run(self, df, trials=100, epochs=100, startstate=0, log=None, logtype='none', seed=-1, progress=None) :
        """ 
        df is properly formatted dataframe of the transition matrix.
        trials is number of trials
        epochs is the maximum number of epochs per trial
        startstate is the starting state in each trial
        log is the log file, None means no log file
        logtype may be 'none' (redundant with log=None) or 'text' or 'csv'
        seed is the random seed
        """
        
        self.log = log 
        self.logtype = logtype
        
        # initialize random
        if seed != -1:
            random.seed(seed)

        # mold the transition table for speed and verify proabilities add to 1
        transtable = df.copy()
        for row in range(len(transtable)) :
            switchvalue = transtable.iloc[row][0]
            for col in range(1,len(transtable)) :
                switchvalue += transtable.iloc[row][col]
                transtable.iloc[row][col] = switchvalue
            if not math.isclose(switchvalue, 1, abs_tol = 0.0001) :
                raise ValueError('Sum of probabilities for transitions should be 1') #Sum of the probabilities should be 1
        
        # find any terminal states
        terminals = [x for x in range(len(df)) if math.isclose(df.iloc[x][x],1,abs_tol = 0.0001)]
        
        # open the log files if needed
        if (log is None) or (logtype == 'none') :
            f = None
        else :
            f = open(log,'wt')
            
        # start the trials loop
        for trial in range(trials) :
            if progress != None :
                progress(trial)
            state = startstate
            for epoch in range(epochs) :
                self._register(trial, epoch, state, f, logtype)
                state = self._transition(state, transtable, terminals)
                if state == -1 :
                    break
        if not (f is None) :
            f.close()

        out = mini_markov.result_set(self.outdf, df)        
        return out
                
    def _register(self, trial, epoch, state, f, logtype) :
        """ 
        trial, epoch, and state are the info to go into the log file f 
        registers the information in that log file as well as the output
        dataframe. If logtype is text, it includes labels. If it is csv, it outputs csv
        """
        if not (f is None):
            if logtype == 'text' :
                f.write(f'Trial {trial} Epoch {epoch}, State {state}\n')
            elif logtype == 'csv' :
                f.write(f'{trial}, {epoch}, {state}\n')
        self.outdf.loc[self._outrow] = [trial, epoch, state]
        self._outrow += 1
        return

    def _transition(self, state, transtable, terminals) :
        """ 
        state is the current state. Uses transition matrix (transtable) to calculate and return next state 
        """
        if state in terminals :
            return -1
        die = random.random()
        col = 0
        while transtable.iloc[state][col] < die :
            col += 1
        return col    


    def build_df(self, columns, transitions) :
        """
        Builds and returns a dataframe properly formatted for this class. columns is a list with the names of
        the states. The transitions is a matrix (array of arrays) of the transitions. 
        """
        d = pd.DataFrame(data=transitions,columns=columns)
        d.index = columns
        return d

    class result_set :
        def __init__(self, dframe, originalframe) :
            """result_set is the output from the class mini_markov

            Args:
                dframe (DataFrame): output dataframe from a mini_markov run
                originalframe (DataFrame): Original transitions dataframe
            """
            self.output = dframe
            self.trials = dframe[['Trial']].nunique()[0] #number of trials
            self.original = originalframe
            self.cols = originalframe.columns
            self.output['Names'] = self.output.apply(lambda row: self.cols[row['State']], axis=1)

        def counts(self) -> tuple : 
            """ Creates and returns frequency data for states

            Returns:
                tuple: two DataFrames. val_counts is the number of events for each state 
                val_counts_per_trial is the mean number of events for each state on a per
                trial basis
            """
            val_counts = self.output['State'].value_counts()
            val_counts.index = self.cols
            val_counts_per_trial = self.output['State'].value_counts()/self.trials
            val_counts_per_trial.index = self.cols
            return val_counts, val_counts_per_trial
        
        def mean_time(self) -> float : 
            """ mean time through the simulation

            Returns:
                float: mean time through the simulation
            """
            return len(self.output)/self.trials
        
        def results_range(self, valsarray = None) -> pd.DataFrame :
            """
            Calculates the min, max, mean, and standard deviation for each state
            Also used for value() with optional valsarray (values array) 

            Returns
            -------
            DataFrame 
                Rows are the states. Columns are Min, Max, Mean, and SD

            """
            if valsarray == None :
                valsarray = np.ones(len(self.cols))
            sums = np.zeros((self.trials, len(self.cols)), dtype=int)
            for epoch in range(len(self.output)) :
                state = self.output.iloc[epoch]['State']
                trial_value = self.output.iloc[epoch]['Trial']
                sums[trial_value, state] += 1*valsarray[state]
            mins = np.amin(sums,axis=0)
            maxs = np.amax(sums,axis=0)
            means = np.mean(sums,axis=0)
            medians = np.median(sums,axis=0)
            stds = np.std(sums,axis=0)
            results = np.vstack((mins,maxs,means,medians,stds))
            idx = ('Min','Max','Mean','Median','StdDev')
            results_df = pd.DataFrame(results,index=idx,columns=self.cols)
            return results_df.transpose()
            
        
        def value(self, valuearray) -> list :
            """
            Parameters
            ----------
            valuearray : list
                list of values for each state

            Returns
            -------
            dataframe with stats from results_range
                product of the valuearray list and the state count for each trial
                (could be considered cost of each trial)

            """
            
            if len(valuearray) != len(self.cols) :
                raise ValueError
            return self.results_range(valsarray=valuearray)    
            
# End of Class mini_markov

# Example code

if __name__ == "__main__":
    import progressbar
    p = progressbar.ProgressBar(maxval=1000).start()
    def callbac(progress):
        p.update(progress)
        return
    tmatrix = [[0.8,0.2,0,0],[0.25,0.5,0.25,0],[0,0.2,0.5,0.3],[0,0,0,1]]
    names = ['Room', 'Hall', 'Lobby', 'Exit']
    m = mini_markov()
    df = m.build_df(names, tmatrix)
    rset = m.run(df,trials=1000,epochs=100,startstate=0,seed=42, log='test.csv', logtype = 'csv',progress=callbac)
    p.finish()

    print(f'Time spent in each state:\n{rset.counts()[0]} \n{rset.counts()[1]}')
    print(f'Average time through system: {rset.mean_time()}')
    print(rset.results_range())
    values = [20,35,30,10]
    print(rset.value(values))
