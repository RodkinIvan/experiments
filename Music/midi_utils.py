from mido import MidiFile, MidiTrack, MetaMessage, Message
import pandas as pd


def get_midi_data(track: MidiTrack):
    df = pd.DataFrame(columns=['on', 'note', 'velocity', 'time'])
    i = 0
    for msg in track:
        if msg.type == 'note_on':
            df.loc[i] = [msg.velocity != 0, msg.note, msg.velocity, msg.time]
            i += 1
            
    return df


def _transform_data(midi_data: pd.DataFrame):
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
        transformed.append(_transform_data(get_midi_data(track)))
    
    transformed = pd.concat(transformed).reset_index()
    transformed = transformed.drop(columns=['index'])

    transformed = transformed.sort_values(by='time')

    transformed['time'] -= pd.concat([pd.Series([0]), transformed['time'][:-1]]).reset_index().drop(columns=['index'])[0]

    return transformed
    

def set_default_prefix(midi_file: MidiFile):
    if len(midi_file.tracks) == 0:
        midi_file.tracks.append(MidiTrack())
    midi_file.tracks[0] = MidiTrack()
    midi_file.tracks[0].append(MetaMessage('track_name', name='PyPiano', time=0))
    midi_file.tracks[0].append(MetaMessage('instrument_name', name='Steinway D Prelude', time=0))
    midi_file.tracks[0].append(MetaMessage('time_signature', numerator=4, denominator=4, clocks_per_click=24, notated_32nd_notes_per_beat=8, time=0))
    midi_file.tracks[0].append(MetaMessage('set_tempo', tempo=500000, time=0))
    midi_file.tracks[0].append(Message('control_change', channel=0, control=64, value=127, time=0))

def midi_from_transformed(t_data: pd.DataFrame) -> MidiFile:
    data = t_data.copy()
    data['time'] = data['time'].cumsum()
    
    note_off = data.copy()
    data.drop(columns=['dur'], inplace=True)

    note_off['time'] += note_off['dur']
    note_off.drop(columns=['dur'], inplace=True)
    note_off['velocity'] = 0

    data = pd.concat([data, note_off]).sort_values('time')

    md = MidiFile()
    set_default_prefix(md)

    prev_time = 0
    for msg in data.iterrows():
        msg = msg[1]
        md.tracks[0].append(Message('note_on', channel=0, note=msg['note'], velocity=msg['velocity'], time=msg['time'] - prev_time))
        prev_time = msg['time']

    md.tracks[0].append(MetaMessage('end_of_track', time=1))
    return md

def load_transformed(t_data: pd.DataFrame, path: str):
    md = midi_from_transformed(t_data)
    md.save(path)