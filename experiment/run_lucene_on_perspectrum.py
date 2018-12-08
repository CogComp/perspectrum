import os

def relevant_perspectives():
    # for any claim, retrieve the relevant perspectives


    for topK in range(5, 50, 2):
        pass


def perspective_stances():
    pass

def perspective_equivalence():
    pass

def supporting_evidences():
    pass

# def experiments():
#     data_dir = "/shared/shelley/khashab2/perspective/data/dataset/perspective_stances/"
#     data_dir_output = data_dir + "output/"
    # train_and_test(data_dir=data_dir, do_train=True, do_eval=True, output_dir=data_dir_output,task_name="Mrpc")

# def evaluation_with_pretrained():
#     bert_model = "/shared/shelley/khashab2/perspective/data/dataset/perspective_stances/output/output.pth"
#     data_dir = "/shared/shelley/khashab2/perspective/data/dataset/perspective_stances/"
#     data_dir_output = data_dir + "output2/"
    # train_and_test(data_dir=data_dir, do_train=False, do_eval=True, output_dir=data_dir_output,task_name="Mrpc",saved_model=bert_model)

if __name__ == "__main__":
    # experiments()
    evaluation_with_pretrained()