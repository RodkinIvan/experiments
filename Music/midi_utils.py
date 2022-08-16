from mido import MidiFile, MidiTrack
import pandas as pd


def get_midi_data(track: MidiTrack):
    df = pd.DataFrame(columns=['on', 'note', 'velocity', 'time'])
    i = 0
    for msg in track:
        if msg.type == 'note_on':
            df.loc[i] = [msg.velocity != 0, msg.note, msg.velocity, msg.time]
            i += 1
            
    return df


def transform_data(midi_data: pd.DataFrame):
    df_ = midi_data.copy()
    df_['time'] = df_['time'].cumsum()
    df_['dur'] = 0
    last_on = [0]*128
    
    i = 0
    for msg in df_.iterrows():
        msg_ = msg[1]
        if msg_['on']:
            last_on[msg_['note']] = i, msg_.time
        else:
            lst = last_on[msg_['note']]
            df_.loc[lst[0], 'dur'] = msg_['time'] - lst[1]
        i += 1

    df_ = df_[df_['dur'] != 0]
    df_ = df_.reset_index()
    df_ = df_.drop(columns=['index', 'on'])

    return df_


def get_transformed_data(midi_file: MidiFile):
    transformed = []
    for track in midi_file.tracks:
        transformed.append(transform_data(get_midi_data(track)))
    
    transformed = pd.concat(transformed).reset_index()
    transformed = transformed.drop(columns=['index'])

    return transformed.sort_values(by='time')
    