from pathlib import Path
import pandas as pd
from .visualization import Visualization
from concurrent.futures import ProcessPoolExecutor

class Experiment:

    def __init__(self,dataset_path: str, excluded_subjects: list = [], excluded_sessions: dict = {}, excluded_trials: dict = {}):
        self.dataset_path = Path(dataset_path)
        self.derivatives_path = Path(str(self.dataset_path) + "_derivatives")
        self.metadata = pd.read_csv(self.dataset_path / "participants.tsv", sep="\t",dtype={"subject_id":str,"old_subject_id":str})
        self.subjects = []
        for subject_id, old_subject_id in zip(self.metadata["subject_id"], self.metadata["old_subject_id"]):
            if subject_id not in excluded_subjects and old_subject_id not in excluded_subjects:
                self.subjects.append(Subject(subject_id, old_subject_id, self, excluded_sessions.get(subject_id,[]), excluded_trials.get(subject_id,{})))


    def get_derivatives_path(self):
        return self.derivatives_path
    
    def get_dataset_path(self):
        return self.dataset_path

    def __iter__(self):
        return iter(self.subjects)
    
    def __getitem__(self, index):
        return self.subjects[index]
    
    def __len__(self):
        return len(self.subjects)
    
    def __repr__(self):
        return f"Experiment = '{self.dataset_path.name}'"
    
    def __next__(self):
        return next(self.subjects)
    
    def load_data(self, detection_algorithm: str,response_column:str = "response",expected_response_column:str = "expected_response",stimulus_column:str = "stimulus"):
        self.detection_algorithm = detection_algorithm
        for subject in self.subjects:
            subject.load_data(detection_algorithm,response_column,expected_response_column,stimulus_column)

    def plot_multipanel(self,display:bool):
        fixations = pd.concat([subject.get_fixations() for subject in self.subjects], ignore_index=True)
        saccades = pd.concat([subject.get_saccades() for subject in self.subjects], ignore_index=True)


        vis = Visualization(self.derivatives_path, self.detection_algorithm)
        vis.plot_multipanel(fixations,saccades,display)

    def filter_fixations(self, min_fix_dur=50, max_fix_dur=1000):
        for subject in self.subjects:
            subject.filter_fixations(min_fix_dur, max_fix_dur)

    def filter_saccades(self, max_sacc_dur=100):
        for subject in self.subjects:
            subject.filter_saccades(max_sacc_dur)

    def drop_trials_with_nan_threshold(self, threshold=0.5):
        bad_trials_total= {}
        for subject in self.subjects:
            bad_sessions,total_sessions,bad_trials = subject.drop_trials_with_nan_threshold(threshold)
            if bad_sessions/total_sessions > threshold:
                # Remove from the list of subjects and delete the subject
                self.subjects.remove(subject)
                del subject
            bad_trials_total[subject.subject_id] = bad_trials
        return bad_trials_total
    
    def plot_scanpaths(self):
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(subject.plot_scanpaths) for subject in self.subjects]
            for future in futures:
                future.result()

    def get_rts(self):
        rts = []
        for subject in self.subjects:
            df = subject.get_rts()
            # Add subject to df
            df["subject_id"] = subject.get_id()
            rts.append(df)
        return pd.concat(rts, ignore_index=True)
        

    def get_subject(self,subject_id):
        for subject in self.subjects:
            if subject.get_id() == subject_id:
                return subject
        return None
    
    def get_session(self,subject_id,session_id):
        subject = self.get_subject(subject_id)
        if subject is not None:
            for session in subject:
                if session.get_id() == session_id:
                    return session
        return None
    
    def get_trial(self,subject_id,session_id,trial_number):
        session = self.get_session(subject_id,session_id)
        if session is not None:
            for trial in session:
                if trial.get_trial_number() == trial_number:
                    return trial
        return None
    
    def get_correct_trials(self):
        correct_trials = {}
        for subject in self.subjects:
            correct_trials[subject.get_id()] = subject.get_correct_trials()
        return pd.DataFrame(correct_trials)
    
    def get_accuracy(self):
        accuracy = []
        for subject in self.subjects:
            accuracy.append((subject.get_id(), subject.get_accuracy()))
        return pd.DataFrame(accuracy, columns=["subject_id","session_id","accuracy"])
class Subject:

    def __init__(self,subject_id: str,old_subject_id: str, experiment: Experiment, excluded_sessions: list = [], excluded_trials: dict = {}):
        self.subject_id = subject_id
        self.old_subject_id = old_subject_id
        self.experiment = experiment
        self.sessions = []
        for session_folder in self.get_subject_path().glob("ses-*"):            
            session_id = session_folder.name.split("-")[-1]
            if session_id not in excluded_sessions:
                self.sessions.append(Session(session_id, self, excluded_trials.get(session_id,{})))


    def get_id(self):
        return self.subject_id

    def get_subject_path_derivatives(self):
        return self.experiment.get_derivatives_path() / f"sub-{self.subject_id}"
    
    def get_subject_path(self):
        return self.experiment.get_dataset_path() / f"sub-{self.subject_id}"
 
    def __iter__(self):
        return iter(self.sessions)
    
    def __getitem__(self, index):
        return self.sessions[index]
    
    def __len__(self):
        return len(self.sessions)
    
    def __repr__(self):
        return f"Subject = '{self.subject_id}', " + self.experiment.__repr__()
    
    def __next__(self):
        return next(self.sessions)
    
    def load_data(self, detection_algorithm: str,response_column:str = "response",expected_response_column:str = "expected_response",stimulus_column:str = "stimulus"):
        self.detection_algorithm = detection_algorithm
        for session in self.sessions:
            session.load_data(detection_algorithm,response_column,expected_response_column,stimulus_column)

    def get_fixations(self):
        return pd.concat([session.get_fixations() for session in self.sessions], ignore_index=True)
    
    def get_saccades(self):
        return pd.concat([session.get_saccades() for session in self.sessions], ignore_index=True)
    
    def filter_fixations(self, min_fix_dur=50, max_fix_dur=1000):
        for session in self.sessions:
            session.filter_fixations(min_fix_dur, max_fix_dur)

    def filter_saccades(self, max_sacc_dur=100):
        for session in self.sessions:
            session.filter_saccades(max_sacc_dur)

    def drop_trials_with_nan_threshold(self, threshold=0.5):
        total_amount = len(self.sessions)
        bad_sessions = 0
        bad_trials_subject = {}
        for session in self.sessions:
            bad_trials, total_trials = session.drop_trials_with_nan_threshold(threshold)
            if len(bad_trials)/total_trials > threshold:
                # Remove from the list of sessions and delete the session
                self.sessions.remove(session)
                del session
                bad_sessions += 1
            bad_trials_subject[session.session_id] = {"bad_trials":bad_trials, "total_trials":total_trials}
        percentage = bad_sessions/total_amount
        if percentage > threshold:
            # Remove all sessions from the list of subjects
            for session in self.sessions:
                self.sessions.remove(session)
                del session
        return bad_sessions, total_amount, bad_trials_subject
    
    def plot_scanpaths(self):
        for session in self.sessions:
            session.plot_scanpaths()

    def get_rts(self):
        rts = []
        for session in self.sessions:
            df = session.get_rts()
            # Add session to df
            df["session_id"] = session.get_id()
            rts.append(df)
        return pd.concat(rts, ignore_index=True)
    
    def get_correct_trials(self):
        correct_trials = {}
        for session in self.sessions:
            correct_trials[session.get_id()] = session.get_correct_trials()
        return pd.DataFrame(correct_trials)

    def get_accuracy(self):
        accuracy = []
        for session in self.sessions:
            accuracy.append(session.get_accuracy())
        return pd.DataFrame(accuracy, columns=["session_id","accuracy"])
    
    def get_session(self,session_id):
        for session in self.sessions:
            if session.get_id() == session_id:
                return session
        return None
    
    def get_trial(self,session_id,trial_number):
        session = self.get_session(session_id)
        if session is not None:
            for trial in session:
                if trial.get_trial_number() == trial_number:
                    return trial
        return None

class Session:
    """
    Initialize a Session instance.

    Args:
        derivatives_path (str): The path to the dataset directory.
        subject_id (str): The subject's ID.
        session_id (str): The session ID.

    Raises:
        FileNotFoundError: If the dataset path or session path does not exist.
    """
    
    def __init__(self, session_id: str, subject:Subject,excluded_trials: list = []):

        self.subject = subject
        self.session_id = session_id
        self.session_path = subject.get_subject_path_derivatives() / f"ses-{self.session_id}"
        self.behavior_path = subject.get_subject_path() / f"ses-{self.session_id}" / "behavioral"
        self.excluded_trials = excluded_trials


        if not self.session_path.exists():
            raise FileNotFoundError(f"Session path not found: {self.session_path}")
            
    def __repr__(self):
      return f"Session = '{self.session_id}', " + self.subject.__repr__()
    
    def drop_trials_with_nan_threshold(self, threshold=0.5):
        """Drop trials with a percentage of NaN values above a certain threshold.

        Args:
            threshold (float): The threshold percentage of NaN values.

        Returns:
            None
        """
        bad_trials = []
        total_trials = len(self.trials)
        for trial in self.trials:
            if trial.is_trial_bad(threshold):
                # Remove from the list of trials and delete the trial
                self.trials.remove(trial)
                bad_trials.append(trial.get_trial_number())
                del trial

        if len(bad_trials)/total_trials > threshold:
            # Remove all trials from the list of sessions
            for trial in self.trials:
                self.trials.remove(trial)
                del trial
        # Return a list with the "trial_number" of the bad trials and the total amount of trials
        return bad_trials, total_trials
        
    def get_id(self):    
        return self.session_id

    def get_rts(self):
        rts = []
        for trial in self.trials:
            df = trial.get_rts()
            rts.append(df)
        return pd.concat(rts, ignore_index=True)
    
    def get_correct_trials(self):
        correct_trials = []
        for trial in self.trials:
            if trial.is_correct():
                correct_trials.append(trial.get_trial_number())
        return pd.Series(correct_trials, name=self.session_id)
    
    def get_accuracy(self):
        return (self.session_id, len(self.get_correct_trials())/len(self.trials))
        
    def load_data(self, detection_algorithm: str,response_column:str = "response",expected_response_column:str = "expected_response",stimulus_column:str = "stimulus"):
        self.detection_algorithm = detection_algorithm
        events_path = self.session_path / f"{self.detection_algorithm}_events"
        self.behavior_data = None
        if self.behavior_path.exists() and len(list(self.behavior_path.glob("*.csv"))) == 1:
            behavior_data = pd.read_csv(next(self.behavior_path.glob("*.csv")))
            self.behavior_data = behavior_data
        elif self.behavior_path.exists() and len(list(self.behavior_path.glob("*.csv"))) > 1:
            raise ValueError(f"Multiple behavior files found in {self.behavior_path}. Please ensure only one behavior file is present.")
    
        # Check if paths and files exist
        if not events_path.exists():
            raise FileNotFoundError(f"Algorithm events path not found: {events_path}")

        # Define file paths
        samples_path = self.session_path  / "samples.hdf5"
        calib_path = self.session_path  / "calib.hdf5"
        header_path = self.session_path  / "header.hdf5"
        msg_path = self.session_path  / "msg.hdf5"
        fix_path = events_path / "fix.hdf5"
        sacc_path = events_path / "sacc.hdf5"
        blink_path = events_path / "blink.hdf5"
        
        # Check if specific files exist
        if not samples_path.exists():
            raise FileNotFoundError(f"Samples file not found: {samples_path}")
        if not fix_path.exists():
            raise FileNotFoundError(f"Fixations file not found: {fix_path}")
        if not sacc_path.exists():
            raise FileNotFoundError(f"Saccades file not found: {sacc_path}")
        if not calib_path.exists():
            raise FileNotFoundError(f"Calibration file not found: {calib_path}")
        if not header_path.exists():
            raise FileNotFoundError(f"Header file not found: {header_path}")
        if not msg_path.exists():
            raise FileNotFoundError(f"Messsages file not found: {msg_path}")
    

        # Load the data
        samples = pd.read_hdf(samples_path)
        fix = pd.read_hdf(fix_path)
        sacc = pd.read_hdf(sacc_path)
        if blink_path.exists():
            blink = pd.read_hdf(blink_path)
        else:
            blink = None
        #self.calib = pd.read_hdf(calib_path)
        #self.header = pd.read_hdf(header_path)
        #self.msg = pd.read_hdf(msg_path)
        self.trials = []
        for trial in [trial for trial in samples["trial_number"].unique() if trial != -1 and trial not in self.excluded_trials]:
            if self.behavior_data is not None:
                behavior_data = self.behavior_data[self.behavior_data["trial_number"] == trial]
                expected_response = behavior_data[expected_response_column].values[0]
                response = behavior_data[response_column].values[0]
                stimulus = behavior_data[stimulus_column].values[0]
            else:
                expected_response = None
                response = None
                stimulus = None
            self.trials.append(Trial(trial,self,samples,fix,sacc,blink,events_path,expected_response,response,stimulus))

    def get_fixations(self):
        return pd.concat([trial.get_fixations() for trial in self.trials], ignore_index=True)
    
    def get_saccades(self):
        return pd.concat([trial.get_saccades() for trial in self.trials], ignore_index=True)
    
    def plot_scanpaths(self,display:bool=False):
        for trial in self.trials:
            trial.plot_scanpath(display=display)

    
    def __iter__(self):
        return iter(self.trials)
    
    def __getitem__(self, index):
        return self.trials[index]
    
    def __len__(self):
        return len(self.trials)
    
    def __next__(self):
        return next(self.trials)
    
    def get_trial(self,trial_number):
        for trial in self.trials:
            if trial.get_trial_number() == trial_number:
                return trial
        return None

    def filter_fixations(self, min_fix_dur=50, max_fix_dur=1000):
        for trial in self.trials:
            trial.filter_fixations(min_fix_dur, max_fix_dur)

    def filter_saccades(self, max_sacc_dur=100):
        for trial in self.trials:
            trial.filter_saccades(max_sacc_dur)

class Trial:
    
    def __init__(self,trial_number:int,session:Session,samples:pd.DataFrame,fix:pd.DataFrame,sacc:pd.DataFrame,blink:pd.DataFrame,events_path:str,expected_response:str,response:str,stimulus:str):
        self.trial_number = trial_number
        self.session = session
        self.samples = samples[samples["trial_number"] == trial_number].reset_index(drop=True)
        self.fix = fix[fix["trial_number"] == trial_number].reset_index(drop=True)
        self.sacc = sacc[sacc["trial_number"] == trial_number].reset_index(drop=True)
        if blink is not None:
            self.blink = blink[blink["trial_number"] == trial_number].reset_index(drop=True)
        self.expected_response = expected_response
        self.response = response
        # The stimulus path should be the path to the image, the folder is named "stimuli" and it is in the same level as the dataset folder
        if stimulus is not None:
            self.stimulus_path = self.session.subject.experiment.dataset_path.parent / "stimuli" / stimulus
        else:
            self.stimulus_path = None
        self.events_path = events_path
        self.detection_algorithm = events_path.name[:-7]

    def get_trial_number(self):
        return self.trial_number
    
    def get_fixations(self):
        return self.fix
    
    def get_saccades(self):
        return self.sacc
    
    def is_correct(self):
        if self.expected_response is None or self.response is None:
            raise ValueError("Expected response or response is None. Please provide the expected response and response from the behavior file.")
        return self.expected_response == self.response
        
    
    def plot_scanpath(self,**kwargs) -> None:
        if not self.events_path.exists():
            raise FileNotFoundError(f"Algorithm events path not found: {self.events_path}")
        vis = Visualization(self.events_path, self.detection_algorithm)
        # Create the plots folder if it does not exist
        (self.events_path / "plots").mkdir(parents=True, exist_ok=True)

        vis.scanpath(fixations=self.fix, saccades=self.sacc, samples=self.samples, screen_height=1080, screen_width=1920,img_path=self.stimulus_path,folder_path=self.events_path/"plots", **kwargs)

    def filter_fixations(self, min_fix_dur=50, max_fix_dur=1000):
        """Filter fixation data by duration and bad data exclusion.

        Args:
            Session

        Returns:
            None
        """
        self.fix = self.fix.loc[(self.fix["duration"] > min_fix_dur) & (self.fix["duration"] < max_fix_dur) & (self.fix["bad"] == False)].reset_index(drop=True)

    def filter_saccades(self, max_sacc_dur=100):
        """Filter saccades data by duration and bad data exclusion.

        Args:
            Session

        Returns:
            None
        """
        self.sacc = self.sacc.loc[(self.sacc["duration"] < max_sacc_dur) & (self.sacc["bad"] == False)].reset_index(drop=True)

    def save_rts(self):
        if hasattr(self, "rts"):
            return
        # Get samples with phase != "", group by"phase" and get the last - first timestamp of each and phase
        rts = self.samples.loc[self.samples["phase"] != ""].groupby(["phase"])["tSample"].agg(lambda x: x.iloc[-1] - x.iloc[0])
        # Rename the column to rt
        self.rts = rts.reset_index().rename(columns={"tSample":"rt"})
        self.rts["trial_number"] = self.trial_number

    def get_rts(self):
        # if rts is not an attribute, save it
        if not hasattr(self, "rts"):
            self.save_rts()
        return self.rts
    

    def is_trial_bad(self, threshold=0.5):
        # Count the amount of NaN values in the samples
        nan_values = self.samples.isna().sum().sum()
        # Count the amount of "bad" values in the samples
        bad_values = self.samples["bad"].sum()

        bad_and_nan = nan_values + bad_values
        # Calculate the percentage of bad values
        bad_and_nan_percentage = bad_and_nan / len(self.samples)

        # Return True if the percentage of bad values is above the threshold
        return bad_and_nan_percentage > threshold