import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import json
import os
from absl import app
from absl import flags
from collections import Counter

flags.DEFINE_string('experiment_name', None, 'name of the experiment')
FLAGS = flags.FLAGS

def check_valid(video_data, start, end):
    if not (video_data['start_jump_2'].item() > 0):
        if end < int(video_data['start_jump_1']):
            return False
        elif start > int(video_data['end_jump_1']):
            return False
    else:
        # before the 1st jump
        if end < int(video_data['start_jump_1']):
            return False
        # inbetween the two jumps
        elif start > int(video_data['end_jump_1']) and end < int(video_data['start_jump_2']):
            return False
        # after the second jump
        elif start > int(video_data['end_jump_2']):
            return False
    return True

def is_overlap(array1, array2):
    return bool(set(array1) & set(array2))

def get_correspond_ans(video_data, start, end):
    for i in range(start, end+1):
        if i in [*range(int(video_data['start_jump_1']), int(video_data['end_jump_1']) + 1)]:
            return (int(video_data['end_jump_1']) - int(video_data['start_jump_1']) + 1) 
    
    if (video_data['start_jump_2'].item() > 0):
        for i in range(start, end+1):
            if i in [*range(int(video_data['start_jump_2']), int(video_data['end_jump_2']) + 1)]:
                return (int(video_data['end_jump_2']) - int(video_data['start_jump_2']) + 1) 
    return 0
def get_err_percentage(video_name, prediction, jump_frame):
    error = 0.0
    num_valid_preds = 0
    num_not_predicted_preds = 0
    
    video_data = jump_frame.loc[jump_frame['Video'] == video_name]
    two_frames = [i for i, p in enumerate(prediction) if p != 0]

    while len(two_frames):
        start_frame = two_frames[0]
        for i, frame in enumerate(two_frames):
            # i is the last frame of the last jump
            if i == (len(two_frames) - 1):
                end_frame = frame
                length = end_frame - start_frame + 1
                if check_valid(video_data, start_frame, end_frame):
                    # print([*range(start_frame, end_frame + 1)])
                    ######### locate the corresponding answer ########
                    correct_length = get_correspond_ans(video_data, start_frame, end_frame)
                    # print(f"correct_length: {correct_length}")
                    # print(f"error: {abs(length - correct_length)}")
                    error += (abs(length - correct_length)/correct_length)
                    num_valid_preds += 1
                
                ## remove this part of list
                two_frames = two_frames[i+1:-1]
                # print(two_frames)
                i = 0
                break
            # i is the last frame of other jumps
            elif two_frames[i+1] != frame + 1:
                end_frame = frame
                length = end_frame - start_frame + 1
                if check_valid(video_data, start_frame, end_frame):
                    # print([*range(start_frame, end_frame + 1)])
                    ######## locate the corresponding answer ########
                    correct_length = get_correspond_ans(video_data, start_frame, end_frame)
                    # print(f"DEBUG{start_frame, end_frame, correct_length})")
                    # print(f"correct_length: {correct_length}")
                    # print(f"error: {abs(length - correct_length)}")
                    error += abs((length - correct_length)/correct_length)
                    num_valid_preds += 1
                
                ## remove this part of list
                two_frames = two_frames[i+1:]
                i = 0
                break
    
    ### Not predicted first jump
    two_frames = [i for i, p in enumerate(prediction) if p != 0]
    end_jump_1 = int(video_data['end_jump_1'])
    start_jump_1 = int(video_data['start_jump_1'])
    first_jump = [i for i in range(start_jump_1+1, end_jump_1)]
    if not (is_overlap(first_jump, two_frames)):
        error += 1
        num_not_predicted_preds += 1
    ### Not predicted second jump
    if (video_data['start_jump_2'].item() > 0):
        end_jump_2 = int(video_data['end_jump_2'])
        start_jump_2 = int(video_data['start_jump_2'])
        second_jump = [i for i in range(start_jump_2+1, end_jump_2)]
        if not is_overlap(second_jump, two_frames):
            error += 1
            num_not_predicted_preds += 1
        
    return  error / (num_valid_preds + num_not_predicted_preds)

def main(_argv):
    # TODO: 哪些影片是“最多” model 表現不好的，從 accuracy & error percentage 來看
    accuracy_dict = {}
    error_percentage_dict = {}
    videos_with_worst_accuracy = []
    videos_with_worst_error = []
    for train_dataset in ["all_jump", "Axel", "Loop", "Flip", "Lutz"]:
        video_list = []
        accuracy_list = []
        error_percentage_list = []
        for test_dataset in ["all_jump"]:
            prediction_file = os.path.join("/home/lin10/projects/SkatingJumpClassifier/experiments", 
                                           FLAGS.experiment_name, 
                                           train_dataset, 
                                           f"{test_dataset}_test_pred.csv")
            predictions = pd.read_csv(prediction_file, header=None, usecols=[0, 1, 2], names=['video_name', 'answer', 'prediction'])
            info_file = f"/home/lin10/projects/SkatingJumpClassifier/data/{test_dataset}/info.csv"
            jump_frame = pd.read_csv(info_file, na_values=["None"])
            print("================================================================")
            print(f"Analyzing {FLAGS.experiment_name}/{train_dataset} model's error")
            print("================================================================")
            for video_name, answer, prediction in zip(predictions['video_name'], predictions['answer'], predictions['prediction']):
                if "Beeper" in video_name:
                    continue
                video_list.append(video_name)
                answer = json.loads(answer)
                prediction = json.loads(prediction)
                accuracy = sum(1 for a,p in zip(answer, prediction) if a == p) / len(answer)
                error_percentage = get_err_percentage(video_name, prediction, jump_frame)
                
                accuracy_list.append(accuracy)
                error_percentage_list.append(error_percentage)
                print(f"{video_name} with accuracy-{accuracy}, error percentage-{error_percentage}")
        
        accuracy_dict[train_dataset] = accuracy_list
        error_percentage_dict[train_dataset] = error_percentage_list
    accuracy_dict["video"] = video_list
    accuracy_df = pd.DataFrame(accuracy_dict)
    accuracy_df.set_index("video")
    error_percentage_dict["video"] = video_list
    error_percentage_df = pd.DataFrame(error_percentage_dict)
    error_percentage_df.set_index("video")
    # print(accuracy_df)      
    # TODO: 每個 model performance 最不好的前十個影片
    for train_dataset in ["all_jump", "Axel", "Loop", "Flip", "Lutz"]:
        print("================================================================")
        print(f"Testing videos with worst accuracy for {train_dataset} model:")
        worst_accuracy_videos = accuracy_df.sort_values(train_dataset)[[train_dataset, 'video']].head(10)
        print(worst_accuracy_videos)
        count_action(worst_accuracy_videos['video'].values.tolist())
        videos_with_worst_accuracy.extend(worst_accuracy_videos['video'].values.tolist())
        
        print(f"Testing videos with worst percentage error for {train_dataset} model:")
        worst_error_videos = error_percentage_df.sort_values(train_dataset)[[train_dataset, 'video']].tail(10)
        print(worst_error_videos)
        count_action(worst_error_videos['video'].values.tolist())
        videos_with_worst_error.extend(worst_error_videos['video'].values.tolist())
        print("================================================================")
    
    # TODO: 計算每個影片的平均 performace，並列出最不好的前十個影片挑出來看
    accuracy_df['mean'] = accuracy_df.mean(axis=1)
    print("Top 10 videos with worst average accuracy")
    print(accuracy_df.sort_values('mean')[['video', 'mean']].head(15))
    
    error_percentage_df['mean'] = error_percentage_df.mean(axis=1)
    print("Top 10 videos with worst average error percentage")
    print(error_percentage_df.sort_values('mean')[['video', 'mean']].tail(15))
    
    
    # TODO: 計算每個model performance 最不好的前十個影片的集合中，每個影片身為表現最不好的次數
    count_video(videos_with_worst_accuracy)
    count_video(videos_with_worst_error)
    
def count_action(video_list):
    for i, video in enumerate(video_list):
        action = video.split('_')[0]
        video_list[i] = action
    counter = Counter(video_list)
    print(counter)
    
def count_video(video_list):
    counter = Counter(video_list)
    print(counter)
    
if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
