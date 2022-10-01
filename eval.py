import pandas as pd
import json
from absl import app
from absl import flags

flags.DEFINE_string('model_name', 'loop_alphapose_42', 'path to model')
flags.DEFINE_string('action', 'loop', 'the name of the testing dataset')
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

def get_correspond_ans(video_data, start, end):
    for i in range(start, end+1):
        if i in [*range(int(video_data['start_jump_1']), int(video_data['end_jump_1']) + 1)]:
            return (int(video_data['end_jump_1']) - int(video_data['start_jump_1']) + 1) 
    
    if (video_data['start_jump_2'].item() > 0):
        for i in range(start, end+1):
            if i in [*range(int(video_data['start_jump_2']), int(video_data['end_jump_2']) + 1)]:
                return (int(video_data['end_jump_2']) - int(video_data['start_jump_2']) + 1) 
    return 0

def get_mae(predictions, jump_frame):
    include_missing_and_false = True
    error = 0
    for video_name, prediction in zip(predictions['video_name'], predictions['prediction']):
        print(video_name)
        prediction = json.loads(prediction)
        video_data = jump_frame.loc[jump_frame['Video'] == video_name]
        two_frames = [i for i, p in enumerate(prediction) if p != 0]
        print(two_frames)
        
        # 計算如果 model 少 predict 的 error:
        if include_missing_and_false:
            if set([*range(int(video_data['start_jump_1']) + 1, int(video_data["end_jump_1"]))]).isdisjoint(two_frames):
                print(video_name, "漏掉 1st jump")
                print(int(video_data["end_jump_1"]) - int(video_data["start_jump_1"]) - 1)
                error += int(video_data["end_jump_1"]) - int(video_data["start_jump_1"]) - 1

            if (video_data['start_jump_2'].item() > 0):
                if set([*range(int(video_data['start_jump_2']) + 1, int(video_data["end_jump_2"]))]).isdisjoint(two_frames):
                    print(video_name, "漏掉 2nd jump")
                    print(int(video_data["end_jump_2"]) - int(video_data["start_jump_2"]) - 1)
                    error += int(video_data["end_jump_2"]) - int(video_data["start_jump_2"]) - 1

        while len(two_frames):
            start_frame = two_frames[0]
            for i, frame in enumerate(two_frames):
                if i == (len(two_frames) - 1):
                    end_frame = frame
                    length = end_frame - start_frame + 1
                    if check_valid(video_data, start_frame, end_frame):
                        print([*range(start_frame, end_frame + 1)])
                        # locate the corresponding answer
                        correct_length = get_correspond_ans(video_data, start_frame, end_frame)
                        error += abs(length - correct_length)
                    elif include_missing_and_false:
                        print([*range(start_frame, end_frame + 1)])
                        error += length
                    
                    ## remove this part of list
                    two_frames = two_frames[i+1:-1]
                    print(two_frames)
                    i = 0
                    break
                elif two_frames[i+1] != frame + 1:
                    end_frame = frame
                    length = end_frame - start_frame + 1
                    if check_valid(video_data, start_frame, end_frame):
                        print([*range(start_frame, end_frame + 1)])
                        # locate the corresponding answer
                        correct_length = get_correspond_ans(video_data, start_frame, end_frame)
                        error += abs(length - correct_length)

                    elif include_missing_and_false:
                        print([*range(start_frame, end_frame + 1)])
                        error += length
                    
                    ## remove this part of list
                    two_frames = two_frames[i+1:]
                    print(two_frames)
                    i = 0
                    break
        
    print("===========================")
    print(f"MAE: {error / len(predictions['video_name'])}")  
    return error / len(predictions['video_name'])

# 計算有與 answer 重疊的 prediction 之誤差百分比
def get_err_percentage(predictions, jump_frame):
    include_missing_and_false = False
    error = 0.0
    num_valid_preds = 0
    for video_name, prediction in zip(predictions['video_name'], predictions['prediction']):
        print(video_name)
        prediction = json.loads(prediction)
        video_data = jump_frame.loc[jump_frame['Video'] == video_name]
        two_frames = [i for i, p in enumerate(prediction) if p != 0]
        # print(two_frames)

        while len(two_frames):
            start_frame = two_frames[0]
            for i, frame in enumerate(two_frames):
                # i is the last frame of the last jump
                if i == (len(two_frames) - 1):
                    end_frame = frame
                    length = end_frame - start_frame + 1
                    if check_valid(video_data, start_frame, end_frame):
                        print([*range(start_frame, end_frame + 1)])
                        # locate the corresponding answer
                        correct_length = get_correspond_ans(video_data, start_frame, end_frame)
                        print(f"correct_length: {correct_length}")
                        print(f"error: {abs(length - correct_length)}")
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
                        print([*range(start_frame, end_frame + 1)])
                        # locate the corresponding answer
                        correct_length = get_correspond_ans(video_data, start_frame, end_frame)
                        print(f"DEBUG{start_frame, end_frame, correct_length})")
                        error += abs((length - correct_length)/correct_length)
                        num_valid_preds += 1
                    
                    ## remove this part of list
                    two_frames = two_frames[i+1:]
                    # print(two_frames)
                    i = 0
                    break

    return num_valid_preds, error / num_valid_preds

def get_custom_error(predictions):
    def get_error(label, tag):
        sett = set([label, tag])
        if label == tag:
            return 0
        elif sett == set([1, 2]) or sett == set([2,3]):
            return 1
        elif sett == set([1, 3]):
            return 2
        elif sett == set([0, 1]) or sett == set([0, 3]):
            return 3
        elif sett == set([0, 2]):
            return 4

    total_error = 0
    for i, prediction in predictions.iterrows():
        video_name = prediction['video_name']
        answer, prediction = json.loads(prediction['answer']), json.loads(prediction['prediction'])
        error = 0
        for label, tag in zip(answer, prediction):
            error += get_error(label, tag)
        total_error += error
        print(f"CUSTOM ERROR of {video_name}: {error}")
    return total_error/ len(predictions)



def main(_argv):
    info_file = f"/home/lin10/projects/SkatingJumpClassifier/data/{FLAGS.action}/info.csv"
    prediction_file = f"/home/lin10/projects/SkatingJumpClassifier/experiments/{FLAGS.model_name}/{FLAGS.action}_test_pred.csv"
    predictions = pd.read_csv(prediction_file, header=None, usecols=[0, 1, 2], names=['video_name', 'answer', 'prediction'])
    jump_frame = pd.read_csv(info_file, na_values=["None"])

    mean_error = get_custom_error(predictions)
    num_valid_preds, err_percentage = get_err_percentage(predictions, jump_frame)
    print("=====================================")
    print(f"# OF VALID PREDS: {num_valid_preds}")
    print(f"MEAN PERCENTAGE ERROR: {err_percentage}")
    print(f"MEAN CUSTOM ERROR: {mean_error}")

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass



