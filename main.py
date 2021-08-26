import matplotlib.pyplot as plt
import util

def main():
    annotators_answer, answer_categorized_by_image_id = util.loadPreprocessAnonymizedProject('./Data/anonymized_project.json')
    
    
    print('---------------------TASK 1.A---------------------\n\n')
    
    
    numberOfAnnotators = util.getNumberOfAnnotators(annotators_answer)
    print('Total number of annotators: {}\n'.format(numberOfAnnotators))
    
    
    print('\n\n---------------------TASK 1.B---------------------\n\n')
    
    
    min_durs_list, ave_durs_list, max_durs_list = util.getMinAveMaxDurations(annotators_answer)
    util.plotMinAveMaxDurations(annotators_answer, min_durs_list, ave_durs_list, max_durs_list) 
    
        
    print('\n\n---------------------TASK 1.C---------------------\n\n')
    
    
    util.printAnnotatorAnswerAmount(annotators_answer)
    
    
    print('\n\n---------------------TASK 1.D---------------------\n\n')
    
    
    util.printYesNoFreq(answer_categorized_by_image_id)    
    
    
    print('\n\n---------------------TASK 2---------------------\n\n')
    
    
    util.printCantSolveCorruptDataTrend(annotators_answer)
    
        
    print('\n\n---------------------TASK 3---------------------\n\n')
    
    references = util.loadReferences('./Data/references.json')
    
    true_false_array = util.getTrueFalseCountFromReferences(references)
    print('Number of True values in References: {}'.format(true_false_array[1]))
    print('Number of False values in References: {}'.format(true_false_array[0]))
    util.plotTrueFalseCountFromReferences(true_false_array)
    
    
    print('\n\n---------------------TASK 4---------------------\n\n')
    
    
    annotators_answer, overall_true_pos, overall_false_pos, overall_false_neg, overall_true_neg, fpr, tpr, roc_auc = util.getPerformanceEvaluation(annotators_answer, references)
    
    # I am writing the overall accuracy here.
    overall_accuracy = (overall_true_pos+overall_true_neg)/(overall_true_pos+overall_false_pos\
                                                            +overall_false_neg+overall_true_neg)
    print('Accuracy of overall answer {}'.format(overall_accuracy))
    
    util.printAccuracyForAnnotators(annotators_answer)
    util.printPrecisionForAnnotators(annotators_answer)
    util.printRecallForAnnotators(annotators_answer)
    util.printF1ScoreForAnnotators(annotators_answer)
    
    util.printAUCForAnnotators(roc_auc)
    util.plotRocCurve(fpr, tpr, roc_auc)
    
if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
    
    