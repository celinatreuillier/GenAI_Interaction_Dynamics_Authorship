import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys


sys.path.append('../scripts_sota/scripts_shibani')


from utils import load_sessions, read_session
from events import generate_event_seq
from summary import stats, print_summary_stats
from main import generate_buffer

def compute_metrics_per_time_window(session_events, time_window=60):
    """
    Splits a session into time windows and computes metrics for each window.
    """
    time_window_metrics = []

    # Convert timestamp to datetime
    session_events['dateTime'] = pd.to_datetime(session_events['eventTimestamp'], unit='ms')

    # Sort events by timestamp
    session_events = session_events.sort_values(by="dateTime")

    # Get the start time of the session
    start_time = session_events["dateTime"].iloc[0]

    # Iterate through time windows
    current_window_start = start_time
    while current_window_start < session_events["dateTime"].iloc[-1]:
        # Define the end of the current time window
        current_window_end = current_window_start + pd.Timedelta(seconds=time_window)

        # Filter events within the current time window
        window_events = session_events[
            (session_events["dateTime"] >= current_window_start) &
            (session_events["dateTime"] < current_window_end)
        ]

        # print(window_events.to_dict('records'))
        text = []
        text_buffer = generate_buffer(window_events.to_dict('records'))
        
        if len(text_buffer) > 0:
            text.append(text_buffer[-1])
            event_seq_dict = generate_event_seq(buffer=text_buffer,
                                                events=window_events.to_dict('records'))
            if event_seq_dict:
                sentence_metrics, api_metrics = stats(event_seq_dict)


                # Compute metrics for the current time window
                num_gpt_calls = api_metrics['Total number of GPT-3 calls made']
                num_used_sugg = api_metrics['Number of times GPT-3 suggestion is used']
                num_modified_sugg = api_metrics['Number of times GPT-3 suggestion is modified']
                num_rejected_sugg = api_metrics['Number of times user rejected GPT-3 suggestion']
                num_used_as_is = api_metrics['Number of times GPT-3 suggestion is used as is']


                total_sentences = sentence_metrics['Total number of sentences']
                gpt_sentences = sentence_metrics['Number of sentences completely authored by GPT-3']
                user_sentences = sentence_metrics['Number of sentences completely authored by the user']
                gpt_user_sentences = sentence_metrics['Number of sentences authored by GPT-3 and user'] + sentence_metrics['Number of sentences of initial prompt'] 
                #To compute gpt_user_sentences, we add the number of sentences in the initial prompt because it is systematically removed in the initial function stats(), and the resulting number if then biased. 

                #Adjust for potential -1 values
                if gpt_user_sentences == -1:
                    gpt_user_sentences = 0
                    user_sentences = user_sentences - 1

                # Avoid division by zero
                if total_sentences > 0:
                    gai_dependence = gpt_sentences / total_sentences
                    writing_autonomy = user_sentences / total_sentences
                    writing_collaboration = gpt_user_sentences / total_sentences
                else:
                    gai_dependence = 0
                    writing_autonomy = 0
                    writing_collaboration = 0

                # Append metrics for the current time window
                time_window_metrics.append({
                    "window_start": current_window_start,
                    "window_end": current_window_end,
                    "nb_gpt_calls": num_gpt_calls,
                    "nb_used_suggestions": num_used_sugg,
                    "nb_modified_suggestions": num_modified_sugg,
                    "nb_rejected_suggestions": num_rejected_sugg,
                    "nb_accepted_as_it_is_suggestions": num_used_as_is,
                    "nb_sentences_total": total_sentences,
                    "nb_sentences_full_GPT": gpt_sentences,
                    "nb_sentences_full_user": user_sentences,
                    "nb_sentences_GPT_user": gpt_user_sentences,
                    "GAI_dependence": gai_dependence,
                    "writing_autonomy": writing_autonomy,
                    "writing_collaboration": writing_collaboration
                })
            else: 
                break
        else:
            break   
    
        # Move to the next time window
        current_window_start = current_window_end
        

    return pd.DataFrame(time_window_metrics)


# Features list = list of metrics that we want to include in the time series
# Output : sessions_list = list of time series, sessions_ids = list of session ids
# A time serie for a session is a 2D array of shape (num_time_windows, num_features)
def prepare_timeseries_data(df, features, session_col='session_id', time_col='time_window'):
    """
    Convertit un DataFrame avec time windows en format tslearn
    """
    # Group by session
    sessions = []
    session_ids = []
    
    for session_id, group in df.groupby(session_col):
        # Sort by time_window
        group = group.sort_values(time_col)
        
        # Features extraction for this session
        session_data = group[features].values
        
        sessions.append(session_data)
        session_ids.append(session_id)
    
    return sessions, session_ids