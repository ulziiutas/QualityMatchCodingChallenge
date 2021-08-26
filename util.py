import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import collections

def loadPreprocessAnonymizedProject(filepath):
    """
        ===============================================
        File load and Preprocessing function for anonymized_project.json
        loadPreprocessAnonymizedProject(filepath)
        ===============================================

        Args:
            filepath - String value for filepath to anonymized_project.json document

        Returns:
            annotators_answer - Dictionary with annotator id as key and answers, cant_solve, corrupt_data as value
            answer_categorized_by_image_id - Dictionary with image id as key and answers for value
    """

    annotator_responses = pd.read_json(filepath)
    
    annotator_responses_dictionary = annotator_responses['results']['root_node']['results']
    
    annotators_answer = dict() # dictionary with annotator id as key and answers, cant_solve, corrupt_data as value

    answer_categorized_by_image_id = dict() # dictionary with image id as key and answers for value

    for key in annotator_responses_dictionary: # I am iterating the keys of the original annotator_responses_dictionary 
         # inside a project input id, the value is all the annotators that has answered for this image
        all_annotators_results = annotator_responses_dictionary[key]['results']

        for single_annotator_result in all_annotators_results: # I iterate through all the annotators that has answered for this image

            annotator_id = single_annotator_result['user']['vendor_user_id']

            if not annotator_id in annotators_answer: # If there is no key in the annotators_answer, I make an empty dictionary
                annotators_answer[annotator_id] = dict()

            # I make a empty list for each of the variables such as node_input_id, image_id, answer, corrupt_data, cant_solve,
            # and duration_ms.

            if not 'node_input_id_list' in annotators_answer[annotator_id]:
                annotators_answer[annotator_id]['node_input_id_list'] = np.array([])

            if not 'image_id' in annotators_answer[annotator_id]:
                annotators_answer[annotator_id]['image_id'] = np.array([])

            if not 'answer_list' in annotators_answer[annotator_id]:
                annotators_answer[annotator_id]['answer_list'] = np.array([])

            if not 'corrupt_data_list' in annotators_answer[annotator_id]:
                annotators_answer[annotator_id]['corrupt_data_list'] = np.array([])

            if not 'cant_solve_list' in annotators_answer[annotator_id]:
                annotators_answer[annotator_id]['cant_solve_list'] = np.array([])

            if not 'duration_ms_list' in annotators_answer[annotator_id]:
                annotators_answer[annotator_id]['duration_ms_list'] = np.array([])

            # After making the empty array value for each key, I append the values to the arrays

            annotators_answer[annotator_id]['node_input_id_list'] = np.append(
                annotators_answer[annotator_id]['node_input_id_list'], 
                single_annotator_result['project_node_input_id'])

            annotators_answer[annotator_id]['answer_list'] = np.append(
                annotators_answer[annotator_id]['answer_list'], 
                single_annotator_result['task_output']['answer'])

            annotators_answer[annotator_id]['corrupt_data_list'] = np.append(
                annotators_answer[annotator_id]['corrupt_data_list'], 
                single_annotator_result['task_output']['corrupt_data'])

            annotators_answer[annotator_id]['cant_solve_list'] = np.append(
                annotators_answer[annotator_id]['cant_solve_list'], 
                single_annotator_result['task_output']['cant_solve'])

            annotators_answer[annotator_id]['duration_ms_list'] = np.append(
                annotators_answer[annotator_id]['duration_ms_list'], 
                single_annotator_result['task_output']['duration_ms'])

            # In order to get only the image id from image url, I used replace and split functions.
            # The image id is usefull to access reference.json file later.
            image_id = single_annotator_result['task_input']['image_url'].replace('.','/').split('/')[-2]

            annotators_answer[annotator_id]['image_id'] = np.append(
                annotators_answer[annotator_id]['image_id'], 
                image_id)

            # I made another dictionary, that uses image id as key and answers for the value which is used to answer 
            # overall questions regarding the images.

            if not image_id in answer_categorized_by_image_id:
                answer_categorized_by_image_id[image_id] = np.array([])

            answer_categorized_by_image_id[image_id] = np.append(
                answer_categorized_by_image_id[image_id],
                single_annotator_result['task_output']['answer'])
    
    annotators_answer = collections.OrderedDict(sorted(annotators_answer.items()))

    return annotators_answer, answer_categorized_by_image_id


def loadReferences(filepath):
    """
        ===============================================
        File load function for references.json
        loadReferences(filepath)
        ===============================================

        Args:
            filepath - String value for filepath to references.json document

        Returns:
            references - Reference table with image id as column and is_bicycle boolean as row
    """

    # I am reading the references.json file and saving it in pandas object
    references = pd.read_json(filepath)

    return references

def getNumberOfAnnotators(annotators_answer):
    """
        ===============================================
        Returns the total number of annotators from annotators_answer dictionary
        getNumberOfAnnotators(annotators_answer)
        ===============================================

        Args:
            annotators_answer - Dictionary with annotator id as key and answers, cant_solve, corrupt_data as value

        Returns:
            numberOfAnnotators - Returns the total number of annotators in annotators_answer dictionary
    """
    
    # Since I used annotator_id as key in annotators_answer dictionary,
    # the total number of keys is same as the number of annotator
    
    numberOfAnnotators = len(annotators_answer.keys())
    
    return numberOfAnnotators
             
def getMinAveMaxDurations(annotators_answer):
    """
        ===============================================
        Returns and Prints the minimum, average, and maximum durations for each annotators
        getMinAveMaxDurations(annotators_answer)
        ===============================================

        Args:
            annotators_answer - Dictionary with annotator id as key and answers, cant_solve, corrupt_data as value

        Returns:
            min_durs_list - Numpy array for each of the annotator's minimum duration
            ave_durs_list - Numpy array for each of the annotator's average duration
            max_durs_list - Numpy array for each of the annotator's maximum duration
    """
    
    # I have made empty arrays in order to store minimum, maximum, and average duration of each annotator.
    min_durs_list = np.array([])
    ave_durs_list = np.array([])
    max_durs_list = np.array([])

    for annotator in annotators_answer: # I am iterating through the annotators

        # I remove the duration values which is less than 0, and stored it in durations_with_only_positive_val
        durations_with_only_positive_val = annotators_answer[annotator]['duration_ms_list']\
                                                            [annotators_answer[annotator]['duration_ms_list']>0]

        # I am calculating minimum, maximum and average values of duration using numpy's function min(), max(), and mean()
        min_durtaion = durations_with_only_positive_val.min()
        ave_durtaion = durations_with_only_positive_val.mean()
        max_duration = durations_with_only_positive_val.max()

        # I am appending the minimum, maximum and average values to the numpy array
        min_durs_list = np.append(min_durs_list, min_durtaion)
        ave_durs_list = np.append(ave_durs_list, ave_durtaion)
        max_durs_list = np.append(max_durs_list, max_duration)

        print('{} max: {} min: {} ave: {}'.format(annotator, max_duration, min_durtaion, ave_durtaion))
    
    return min_durs_list, ave_durs_list, max_durs_list

def plotMinAveMaxDurations(annotators_answer, min_durs_list, ave_durs_list, max_durs_list):      
    """
        ===============================================
        Plots the minimum, average, and maximum durations for each annotators
        plotMinAveMaxDurations(annotators_answer, min_durs_list, ave_durs_list, max_durs_list)
        ===============================================

        Args:
            annotators_answer - Dictionary with annotator id as key and answers, cant_solve, corrupt_data as value
            min_durs_list - Numpy array for each of the annotator's minimum duration
            ave_durs_list - Numpy array for each of the annotator's average duration
            max_durs_list - Numpy array for each of the annotator's maximum duration

        Returns:
            Plots the minimum, average, and maximum durations for each annotators
    """
    
    # Here I am plotting the minimum, maximum and average values for each annotators
    fig, axs = plt.subplots(3, sharex=True, figsize=(20, 10))

    axs[0].bar(np.arange(len(annotators_answer)), min_durs_list)
    axs[0].set_ylabel("Duration(ms)", size=14)
    axs[0].set_title('Minimum Duration', size=16)

    axs[1].bar(np.arange(len(annotators_answer)), ave_durs_list)
    axs[1].set_ylabel("Duration(ms)", size=14)
    axs[1].set_title('Average Duration', size=16)

    axs[2].bar(np.arange(len(annotators_answer)), max_durs_list)
    axs[2].set_ylabel("Duration(ms)", size=14)
    axs[2].set_title('Maximum Duration', size=16)

    plt.xticks(np.arange(len(annotators_answer)), list(annotators_answer.keys()), rotation='vertical')
    plt.xlabel("Annotator ID", size=14)
    plt.show()
    
def printAnnotatorAnswerAmount(annotators_answer):   
    """
        ===============================================
        Prints the total amount of answers for each annotators
        printAnnotatorAnswerAmount(annotators_answer)
        ===============================================

        Args:
            annotators_answer - Dictionary with annotator id as key and answers, cant_solve, corrupt_data as value

        Returns:
            Prints the total amount of answers for each annotators
    """
    
    # I am iterating through the annotars and print the length of answers for each annotator
    for annotator in annotators_answer:
        print("Amount of results for",annotator," :",len(annotators_answer[annotator]['node_input_id_list']))  

def printYesNoFreq(answer_categorized_by_image_id):
    """
        ===============================================
        Prints frequency of Yes and No answers for each images by all annotators
        printYesNoFreq(answer_categorized_by_image_id)
        ===============================================

        Args:
            answer_categorized_by_image_id - Dictionary with image id as key and answers for value

        Returns:
            Prints frequency of Yes and No answers for each images by all annotators
    """

    for image_id in answer_categorized_by_image_id:

        # I find the number of no answers and divide by total number of answers to get the frequency of answer NO
        freq_of_no_answer = np.sum(answer_categorized_by_image_id[image_id] == 'no')\
                                /len(answer_categorized_by_image_id[image_id]) 
        # By subtracting from 1, we can get the frequency of YES answer
        freq_of_yes_answer = 1 - freq_of_no_answer

        # The meaning of disagreeing is that the annotators have answer NO and YES almost equally 
        # otherwise it would mean they agree mostly.
        if abs(freq_of_no_answer - freq_of_yes_answer) <= 0.2:
            print("Image ID: ", image_id, "Frequency of NO answer: ", freq_of_no_answer)
            print("Image ID: ", image_id, "Frequency of YES answer: ", freq_of_yes_answer,'\n')

def printCantSolveCorruptDataTrend(annotators_answer):
    """
        ===============================================
        Prints the trend between Cant Solve and Corrupt Data options for each annnotators
        printCantSolveCorruptDataTrend(annotators_answer)
        ===============================================

        Args:
            annotators_answer - Dictionary with annotator id as key and answers, cant_solve, corrupt_data as value

        Returns:
            Prints the trend between Cant Solve and Corrupt Data options for each annnotators
    """
   
    for annotator in annotators_answer: # I am iterating through the annotators
    
        # I am finding the indeces where either the cant_solve or corrupt_data options are used.
        intersection_cant_solve_corrupt_data = np.logical_or(annotators_answer[annotator]['cant_solve_list'] == 1,\
                                                             annotators_answer[annotator]['corrupt_data_list'] == 1)

        # I have split the cases where either the cant_solve or corrupt_data options are used
        corrupt_data_list_split = annotators_answer[annotator]['corrupt_data_list'][intersection_cant_solve_corrupt_data]
        cant_solve_list_split = annotators_answer[annotator]['cant_solve_list'][intersection_cant_solve_corrupt_data]


        if len(cant_solve_list_split) != 0: 
            # if there is use case of either options, we find the matching cases of the two options

            simple_matching = np.sum(cant_solve_list_split == corrupt_data_list_split)/len(cant_solve_list_split)

            print("{} - Trend(cant solve-corrupt data): {}\n".format(annotator, simple_matching))

        else: # if there is no use case of either options, I print a message
            print("{} - didn't use cant_solve or corrupt_data options\n".format(annotator))
               
def getTrueFalseCountFromReferences(references):
    """
        ===============================================
        Returns numpy array with [Number of False answers, Number of True Answers] of the Reference table
        getTrueFalseCountFromReferences(references)
        ===============================================

        Args:
            references - Reference table with image id as column and is_bicycle boolean as row

        Returns:
            true_false_array - Numpy array with two values which are [Number of False answers, Number of True Answers]
    """
   
    number_of_False = 0
    number_of_True = 0

    for item in references: # I am iterating each images

        if references[item].is_bicycle: # If the value is TRUE, I increment number_of_True
            number_of_True += 1
        else: # If the value is FALSE, I increment number_of_False
            number_of_False += 1

    # I have put the number_of_False, number_of_True values inside numpy array
    true_false_array = np.array([number_of_False, number_of_True])
               
    return true_false_array

def plotTrueFalseCountFromReferences(true_false_array):
    """
        ===============================================
        Plots bar plot for numpy array with [Number of False answers, Number of True Answers] of the Reference table
        plotTrueFalseCountFromReferences(true_false_array)
        ===============================================

        Args:
            true_false_array - Numpy array with two values which are [Number of False answers, Number of True Answers]

        Returns:
            Plots bar plot for numpy array with [Number of False answers, Number of True Answers] of the Reference table
    """
   
    fig, ax = plt.subplots(figsize = (6,8))
    ax.bar(np.arange(2), true_false_array)
    plt.xticks(np.arange(2), ['False Answer', 'True Answer'], rotation='vertical')
    for index,data in enumerate(true_false_array):
        plt.text(x=index-0.15 , y =data+20 , s=f"{data}" , fontdict=dict(fontsize=20))    
    plt.xlabel("Is_Bycicle", size=14)
    plt.ylabel("Number of answer", size=14)
    plt.title("False vs True Answer", size=16)
    plt.show()

def getPerformanceEvaluation(annotators_answer, references):
    """
        ===============================================
        Returns overall True Positive, False Positive, False Negative, True Negative values of all annotators, return individual True Positive, False Positive, False Negative, True Negative values as modified annotators_answer, calculates and returns fpr, tpr, and roc_auc values using sklearn
        getPerformanceEvaluation(annotators_answer, references)
        ===============================================

        Args:
            annotators_answer - Dictionary with annotator id as key and answers, cant_solve, corrupt_data as value
            references - Reference table with image id as column and is_bicycle boolean as row

        Returns:
            annotators_answer - Same Dictionary from args but added true_pos, false_pos, false_neg, true_neg values for each annotators
            overall_true_pos - Overall True Positive value of all annotators
            overall_false_pos - Overall False Positive value of all annotators
            overall_false_neg - Overall False Negative value of all annotators
            overall_true_neg - Overall True Negative value of all annotators
            fpr - False positive rate calculated from sklearn's roc_curve function
            tpr - True positive rate calculated from sklearn's roc_curve() function
            roc_auc - AUC value calculated from sklearn's auc() function
    """

    # to calculate overall accuracy, I am using the below variables
    overall_true_pos = 0
    overall_false_pos = 0
    overall_false_neg = 0
    overall_true_neg = 0

    # I made empty dictionary to find ROC and AUC for each annotator
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for annotator in annotators_answer:
        true_pos = 0
        false_pos = 0
        false_neg = 0
        true_neg = 0

        # I am using an numpy array to restore answer and corresponding reference values
        annotators_answer[annotator]['answered_values'] = np.array([])
        annotators_answer[annotator]['reference_values'] = np.array([])

        # I iterate through the annotators answers with value and index 
        for idx, answer in enumerate(annotators_answer[annotator]['answer_list'], start=0):


            if answer == 'yes':
                # if the answer is YES, I save a TRUE value into the numpy list
                annotators_answer[annotator]['answered_values'] = np.append(annotators_answer[annotator]['answered_values']\
                                                                           ,True)

                # if the reference is also TRUE, then I increase true positive value otherwise I increase false positive
                if references[annotators_answer[annotator]['image_id'][idx]].is_bicycle:
                    true_pos += 1
                else:
                    false_pos += 1

            else:
                # if the answer is NO, I save a FALSE value into the numpy list
                annotators_answer[annotator]['answered_values'] = np.append(annotators_answer[annotator]['answered_values']\
                                                                           ,False)

                # if the reference is also FALSE, then I increase true negative value otherwise I increase false negative
                if not references[annotators_answer[annotator]['image_id'][idx]].is_bicycle:
                    true_neg += 1                
                else:
                    false_neg += 1

            # I add the reference value into a numpy array
            annotators_answer[annotator]['reference_values'] = np.append(annotators_answer[annotator]['reference_values'],\
                                                                 references[annotators_answer[annotator]['image_id'][idx]].is_bicycle)

        # I am using sklearn's roc_curve function to find the false positive rate and true positive rate
        fpr[annotator], tpr[annotator], _ = roc_curve(annotators_answer[annotator]['reference_values'],\
                                                      annotators_answer[annotator]['answered_values'])

        # using the sklearn's auc function I am area under the ROC.
        roc_auc[annotator] = auc(fpr[annotator], tpr[annotator])

        annotators_answer[annotator]['true_pos'] = true_pos
        annotators_answer[annotator]['false_pos'] = false_pos
        annotators_answer[annotator]['true_neg'] = true_neg
        annotators_answer[annotator]['false_neg'] = false_neg

        overall_true_pos += true_pos
        overall_false_pos += false_pos
        overall_true_neg += true_neg
        overall_false_neg += false_neg
                       
    return annotators_answer, overall_true_pos, overall_false_pos, overall_false_neg, overall_true_neg, fpr, tpr, roc_auc

def printAccuracyForAnnotators(annotators_answer):
    """
        ===============================================
        Prints the Accuracy of each annotators
        printAccuracyForAnnotators(annotators_answer)
        ===============================================

        Args:
            annotators_answer - Dictionary with annotator id as key and answers, cant_solve, corrupt_data, added true_pos, false_pos, false_neg, true_neg as value

        Returns:
            Prints the Accuracy of each annotators
    """
   
    for annotator in annotators_answer: 
        accuracy = (annotators_answer[annotator]['true_pos']+annotators_answer[annotator]['true_neg'])\
        /(annotators_answer[annotator]['true_pos']+annotators_answer[annotator]['false_pos']\
          +annotators_answer[annotator]['true_neg']+annotators_answer[annotator]['false_neg'])

        print('Accuracy of {} : {}'.format(annotator, accuracy))
    print('\n')
               
def printPrecisionForAnnotators(annotators_answer):
    """
        ===============================================
        Prints the Precision of each annotators
        printPrecisionForAnnotators(annotators_answer)
        ===============================================

        Args:
            annotators_answer - Dictionary with annotator id as key and answers, cant_solve, corrupt_data, added true_pos, false_pos, false_neg, true_neg as value

        Returns:
            Prints the Precision of each annotators
    """
               
    for annotator in annotators_answer: 
        precision = annotators_answer[annotator]['true_pos']\
        /(annotators_answer[annotator]['true_pos']+annotators_answer[annotator]['false_pos'])

        print('Precision of {} : {}'.format(annotator, precision))
    print('\n')

def printRecallForAnnotators(annotators_answer):
    """
        ===============================================
        Prints the Recall of each annotators
        printRecallForAnnotators(annotators_answer)
        ===============================================

        Args:
            annotators_answer - Dictionary with annotator id as key and answers, cant_solve, corrupt_data, added true_pos, false_pos, false_neg, true_neg as value

        Returns:
            Prints the Recall of each annotators
    """

    for annotator in annotators_answer: 
        recall = (annotators_answer[annotator]['true_pos'])\
        /(annotators_answer[annotator]['true_pos']+annotators_answer[annotator]['false_neg'])

        print('Recall of {} : {}'.format(annotator, recall))
    print('\n')

def plotRocCurve(fpr, tpr, roc_auc):
    """
        ===============================================
        Plots the ROC curve and AUC values of each annotators
        plotRocCurve(fpr, tpr, roc_auc)
        ===============================================

        Args:
            fpr - False positive rate calculated from sklearn's roc_curve function
            tpr - True positive rate calculated from sklearn's roc_curve() function
            roc_auc - AUC value calculated from sklearn's auc() function

        Returns:
            Plots the ROC curve and AUC values of each annotators
    """
               
    fig, axs = plt.subplots(11, 2, sharex=True, sharey=True, figsize=(20, 150))

    row_idx = 0
    col_idx = 0

    for annotator in fpr:
        lw = 2

        axs[row_idx, col_idx].plot(fpr[annotator], tpr[annotator], color='darkorange',
                 lw=lw, label='ROC curve (area = %0.4f)' % roc_auc[annotator])
        axs[row_idx, col_idx].plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        axs[row_idx, col_idx].legend(loc="lower right")
        axs[row_idx, col_idx].set_title(annotator)

        if col_idx == 0:
            col_idx += 1
        else:
            col_idx = 0    
            row_idx += 1
               
    plt.show()
               
               
               
               
               
               
               
               
               
               
               
               