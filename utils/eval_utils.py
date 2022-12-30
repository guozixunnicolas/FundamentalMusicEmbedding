import numpy as np
from collections import Counter
interval_dict = {0: "unison", 1:"2nd", 2:"2nd", 3:"3rd", 4:"3rd", 5:"4th", 6:"tt", 7:"5th", 8:"6th", 9:"6th", 10:"7th", 11:"7th"}
dur_dict = {0: 'pad', 1: 0.25, 2: 0.5, 3: 0.75, 4: 1.0, 5: 1.25, 6: 1.5, 7: 1.75, 8: 2.0, 9: 2.25, 10: 2.5, 11: 2.75, 12: 3.0, 13: 3.25, 14: 3.5, 15: 3.75, 16: 4.0}

			
pitch_lst = [x for x in range(3, 131)]
dur_lst = [x for x in range(1, 17)]
# def get_n_grams(inp, n_gram = 4):
def get_rep_seq(inp, n_gram = 4):
    """per song feature"""
    gram_lst= [tuple(inp[i:i+n_gram]) for i in range(len(inp)-n_gram+1)]

    unique_gram = list(set(gram_lst))
    return 1.0 - len(unique_gram)/len(gram_lst)

def get_unique_tokens(inp, tpe = "pitch"):
    """aggregate and normalize"""
    if tpe=="pitch":
        counts = [inp.count(keys) for keys in pitch_lst] #list with length 12, each count number of intervals
    elif tpe=="dur":
        counts = [inp.count(keys) for keys in dur_lst] #list with length 12, each count number of intervals

    return counts

def get_unique_intervals(inp, banned_tokens = [0, 1, 2]):
    """aggregate and normalize"""
    inp = [x for x in inp if x not in banned_tokens]
    inp = np.array(inp)
    inp_diff = abs(np.diff(inp))
    counts = [inp_diff.tolist().count(keys) for keys,_ in interval_dict.items()] #list with length 12, each count number of intervals
    return counts    

def get_arpeggio_num(inp_pitch, inp_dur, sliding_window_size = 4, note_distance_threshold = 5, banned_tokens = [0, 1, 2]):
    gram_lst= [tuple(inp_pitch[i:i+sliding_window_size]) for i in range(len(inp_pitch)-sliding_window_size+1)]
    gram_lst_dur= [tuple(inp_dur[i:i+sliding_window_size]) for i in range(len(inp_pitch)-sliding_window_size+1)]

    num_of_arpegio = 0
    for gram, gram_dur in zip(gram_lst, gram_lst_dur):
        inp_diff = np.diff(gram)
        if_arpegio_criteria0 = not any([x in banned_tokens for x in gram]) # rest, sustain, pad should not be in the list
        if_arpegio_criteria1 = all(inp_diff>0) or all(inp_diff<0) # monotonically increasing/ decreasing
        if_arpegio_criteria2 = all(abs(inp_diff)<note_distance_threshold)# small interval
        outlier_dur_num = len(gram_dur)-len(list(set(gram_dur)))
        # print("huh?",gram_dur, list(set(gram_dur)))
        if_arpegio_criteria3 = len(list(set(gram_dur)))<=2# similar duration

        if if_arpegio_criteria0 and if_arpegio_criteria1 and if_arpegio_criteria2 and if_arpegio_criteria3:
            num_of_arpegio+=1
        # print(gram, inp_diff,if_arpegio_criteria0,if_arpegio_criteria1, if_arpegio_criteria2, if_arpegio_criteria3)

    # arpeggio_ratio = num_of_arpegio/len(gram_lst)

    # # print(gram_lst[:3])

    return num_of_arpegio, len(gram_lst)

def get_num_empty_bars(inp_pitch, inp_dur):
    total_empty_dur_lst = []
    for pos, (note, dur) in enumerate(zip(inp_pitch, inp_dur)): 

        if note==1:
            total_empty_dur_lst.append(dur)
            #check future pos whether sustain
            next_pos = pos + 1 
            if next_pos<=len(inp_pitch)-1:
                note_next, dur_next = inp_pitch[next_pos], inp_dur[next_pos]
                while(note_next == 2 and next_pos<=len(inp_pitch)-1):
                    total_empty_dur_lst.append(dur_next)
                    # dur += dur_next
                    next_pos+=1
                    try:
                        note_next, dur_next = inp_pitch[next_pos], inp_dur[next_pos]
                    except:
                        note_next = "break the loop"
                        dur_next = "break the loop"
    empty_dur = sum([dur_dict[x] for x in total_empty_dur_lst])
    num_empty_bars = empty_dur/4
    return num_empty_bars
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
if __name__ =="__main__":
    # inp_pitch = [1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    # inp_dur = [16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16]
    inp_pitch = [1, 60, 60, 63, 60, 58, 55, 53, 55, 56, 53, 55, 53, 51, 1, 1, 65, 67, 65, 60, 63, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    inp_dur = [4, 8, 1, 1, 1, 1, 12, 2, 1, 1, 4, 2, 1, 1, 1, 2, 1, 4, 8, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 4, 1, 1, 4, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 4, 4, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    

    num_empty_bars = get_num_empty_bars(inp_pitch, inp_dur)
    print(num_empty_bars)
    # inp = [63, 75, 75, 75, 75, 75, 75, 63, 75, 75, 75, 73, 72, 70, 68, 67, 68, 70, 63, 72, 73, 1, 74, 72, 75, 74, 70, 67, 68, 70, 72, 74, 75, 74, 72, 70, 68, 1, 70, 68, 70, 74, 72, 70, 65, 1, 72, 72, 70, 67, 75, 75, 74, 75, 77, 75, 74, 72, 70, 68, 67, 1, 74, 75, 77, 77, 70, 63, 1, 75, 75, 75, 74, 72, 70, 72, 1, 72, 75, 79, 77, 77, 79, 82, 75, 74, 72, 70, 68, 79, 75, 79, 80, 79, 1, 75, 77, 79, 1, 70, 68, 67, 1, 70, 74, 72, 1, 75, 79, 1, 77, 75, 74, 1, 75, 74, 75, 77, 75, 1, 77, 1, 79, 80, 1, 72, 1, 75, 1, 75, 1, 77, 79, 80, 79, 1, 77, 75, 75, 75, 1, 74, 75, 1, 77, 1, 77, 75, 77, 79, 82, 80, 79, 75, 75, 1, 82, 84, 82, 79, 1, 75, 72]
    # inp_pitch = inp 
    # inp_dur = [1, 1, 1, 1, 2, 1, 1, 2, 2, 2, 2, 3, 1, 2, 2, 3, 1, 4, 2, 2, 4, 2, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 2, 2, 2, 4, 4, 4, 2, 4, 2, 4, 2, 2, 2, 1, 1, 4, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 4, 4, 4, 1, 4, 2, 2, 1, 1, 1, 1, 1, 4, 1, 1, 4, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 4, 6, 4, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    n_gram = 4
    sliding_window_size = 5
    num_arpeggio, total_grams = get_arpeggio_num(inp_pitch, inp_dur)
    print(num_arpeggio, total_grams)
    

