def getFramesList(txtFile, start_context, context_with_issue):
    context_name_list = []
    frames_dict = {}
    list_ = open(txtFile).read().split()
    start_fetching = False
    for line in list_:
        frames_line = line.split(',')
        context_name, time_stamp = frames_line
        if(context_name == start_context):
            start_fetching = True
        if(not start_fetching):
            continue
        if not(context_name in context_name_list) and not(context_name in context_with_issue):
            context_name_list.append(context_name)
        if not(context_name in frames_dict):
            frames_dict[context_name] = [time_stamp]
        else:
            frames_dict[context_name].append(time_stamp)

    for context_name in frames_dict:
        time_stamp_list = frames_dict[context_name]
        frames_dict[context_name] = sorted(time_stamp_list)
    #     print("======", context_name)
    #     print(len(frames_dict[context_name]))
    #     for time in frames_dict[context_name]:
    #         print(time)
    return context_name_list, frames_dict

context_with_issue = [
    ## only at test: 
    # "10980133015080705026_780_000_800_000"

]
start_context = "10149575340910243572_2720_000_2740_000" # only append data starting from this context
txtFile = "./2d_pvps_test_frames.txt"
context_name_list, frames_dict = getFramesList(txtFile, start_context, context_with_issue)
# for key in frames_dict:
#     print(key)

for c in context_name_list:
    print(c)
print(len(context_name_list))


import os
path = "/media/kaiwenjon/Kevin-linux-dats/waymo/dataset_jpg/testing"
directories = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
print(len(directories))