# -- Imports --
import numpy as np
import pandas as pd
from tqdm import tqdm

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.core.common.SettingWithCopyWarning)

from iou_3d_functions import compute_box_3d_pandas
from iou3d import iou3d

from scipy.optimize import linear_sum_assignment
    
# -- Main function --
def main(args=None):

    # -- Read and load groundtruth and detection data for evaluation.
    # GT_FILE = "groundtruth/01.csv"
    # # EVAL_FILE = "groundtruth/01.csv"
    # EVAL_FILE ="detections/eval_itsc_01.csv"
    # GT_FILE = "detections.csv"
    # EVAL_FILE = "detections.csv"
    SCENARIO = "06_town07_lluvia"
    SENSOR = "radar"
    GT_FILE = "examples/gt_" + SCENARIO + ".csv"
    EVAL_FILE = "examples/det_" + SCENARIO  + ".csv"

    DISTANCE = 75

    df_groundtruth = pd.read_csv(GT_FILE)
    df_detections = pd.read_csv(EVAL_FILE)

    # -- Select the categories to evaluate.
    # CATEGORIES = ["Unknown", "Unknown_Small","Unknown_Medium","Unknown_Big","Pedestrian", "Bike","Car", "Truck","Motorcycle", "Other_Vehicle","Barrier", "Sign"]
    CATEGORIES = ["Car"]

    # -- TODO: Select metrics to evaluate.
    # METRICS = ["precision", "recall", "IoU", "VE"]

    # -- Thresholds for evaluation.
    IOU_THRESHOLD = 0.01
    DIST_THRESHOLD = 3
    SCORE_THRESHOLD = np.arange(0, 1.1, 0.05)
    TIMESTAMP_RANGE = 0.25

    # -- Add a column to store the original indices.
    df_groundtruth["original_index"] = df_groundtruth.index
    df_detections["original_index"] = df_detections.index


    # -- Add a column to both dataframes to indicate if processed.
    df_groundtruth['processed'] = False
    df_detections['processed'] = False

    print(f"Length of the original groundtruth df: {len(df_groundtruth)}")

    # -- Filter groundtruth data to customise evaluation.
    # By default, the evaluation should be done on all groundtruth data,
    # but in this case we filter to the frontal field of view.
    FOV = 90
    df_groundtruth = df_groundtruth[np.arctan2(df_groundtruth['y'], df_groundtruth['x']) >= -FOV/2 * np.pi/180]
    df_groundtruth = df_groundtruth[np.arctan2(df_groundtruth['y'], df_groundtruth['x']) < FOV/2 * np.pi/180]

    df_detections = df_detections[np.arctan2(df_detections['y'], df_detections['x']) >= -FOV/2 * np.pi/180]
    df_detections = df_detections[np.arctan2(df_detections['y'], df_detections['x']) < FOV/2 * np.pi/180]

    # -- Filter groundtruth distance.
    df_groundtruth = df_groundtruth[np.sqrt(df_groundtruth['x']**2 + df_groundtruth['y']**2) <= DISTANCE]
    df_detections = df_detections[np.sqrt(df_detections['x']**2 + df_detections['y']**2) <= DISTANCE]


    # -- Easy evaluation filters: KITTI.
    # df_groundtruth = df_groundtruth[df_groundtruth['bottom'] - df_groundtruth['top'] > 40]
    # df_groundtruth = df_groundtruth[df_groundtruth['occluded'] <= 1]

    print(f"Length of the filtered-by-FoV groundtruth df: {len(df_groundtruth)}")

    # -- Round timestamps to 3 decimal places.
    df_groundtruth['timestamp'] = df_groundtruth['timestamp'].round(3)

    # -- Add a column to both dataframes with its bounding box.
    df_groundtruth['bbox'] = df_groundtruth.apply(compute_box_3d_pandas, axis=1)
    df_detections['bbox'] = df_detections.apply(compute_box_3d_pandas, axis=1)

    # -- Create the evaluation dataframe.
    df_results = pd.DataFrame(columns=['timestamp', 'status', 'iou'])

    # -- Round the timestamps to 3 decimal places.
    df_groundtruth['timestamp'] = df_groundtruth['timestamp'].round(3)
    df_detections['timestamp'] = df_detections['timestamp'].round(3)

    tp, fp, fn = 0, 0, 0
    # -- Iterate through all categories.
    for category in CATEGORIES:

        # -- Filter groundtruth and detections to current category.
        df_gt_category = df_groundtruth[df_groundtruth['type'] == category]
        df_det_category = df_detections[df_detections['type'] == category]
        print(f"Length of the category {category} groundtruth df: {len(df_gt_category)}")

        # -- Iterate through all frames and create auxiliary progress bar to monitor progress.
        unique_frames = df_gt_category['frame'].unique()
        progress_bar = tqdm(total=len(unique_frames))
        for frame in unique_frames:

            # -- Associate groundtruth and detections to current frame according to timestamp.
            df_gt_frame = df_gt_category[df_gt_category['frame'] == frame]
            df_det_frame = df_det_category[df_det_category['timestamp'] == df_gt_frame['timestamp'].iloc[0]]

            df_gt_frame_copy, df_det_frame_copy = df_gt_frame.copy(), df_det_frame.copy()

            # -- Create cost matrix for current frame.
            n_gt_objs = len(df_gt_frame_copy)
            n_det_objs = len(df_det_frame_copy)
            cost_matrix = []
            if n_gt_objs != 0 and n_det_objs != 0:
                cost_matrix = df_gt_frame_copy.apply(
                    lambda row_gt: df_det_frame_copy.apply(
                        lambda row_det: iou3d(row_gt['bbox'], row_det['bbox']), axis=1).tolist(),
                    axis=1).tolist()

            if n_gt_objs != n_det_objs or cost_matrix == []:
                if n_gt_objs < n_det_objs:
                    cost_matrix += [[0] * n_det_objs] * (n_det_objs - n_gt_objs)
                else:
                    cost_matrix = list(map(lambda row_gt: row_gt + [0] * (n_gt_objs - n_det_objs), cost_matrix))

                    
            

            # -- Apply linear assignment to current frame.
            assignment = ([],[])
            row_gt, col_det = ([], [])

            # print("==================================================================")
            if cost_matrix != []:
                row_gt, col_det = linear_sum_assignment(cost_matrix, maximize=True)
                assignment = linear_sum_assignment(cost_matrix, maximize=True)
                # print(f"Row col: {row_gt}, {col_det}")

            # print(cost_matrix)
            # print(f"GT Objs: {n_gt_objs}, Det Objs: {n_det_objs}")

            # -- Create assignment dataframe.
            # df_assignment = pd.DataFrame(columns=['gt_index', 'det_index'])
            # df_assignment['gt_index'] = row_gt
            # df_assignment['det_index'] = col_det
            # print(df_assignment)

            for idx, element in enumerate(row_gt):
                if cost_matrix[idx][col_det[element]] > 0.05:
                    tp += 1
                    # print(f"{element} is TP.")
                elif cost_matrix[idx][col_det[element]] == 0 and idx < n_gt_objs - 1:
                    fn += 1
                    # print(f"{element} is FN.")
                else:
                    fp += 1
                    # print(f"{element} is FP.")


            # print("==================================================================")


            # -- Iterate through all groundtruth objects in current frame.
            for gt_index, gt_row in df_gt_frame.iterrows():

                # -- Filter detections to 4m of distance.
                df_det_candidates = df_det_frame[(np.square(df_det_frame['x'] - gt_row['x']) + np.square(df_det_frame['x'] - gt_row['x'])) ** (0.5) < DIST_THRESHOLD]
                # print(f"There are {len(df_det_candidates)} detections in the vicinity of the groundtruth object.")

                CANDIDATES = len(df_det_candidates) != 0

                # -- If there are candidates, iterate through them.
                if CANDIDATES:

                    # -- Calculate Intersection over Union (IoU) for current groundtruth object and all candidates.
                    df_det_candidates['iou'] = df_det_candidates.apply(lambda x: iou3d(x['bbox'], gt_row['bbox']), axis=1)
                    # print(df_det_candidates['iou'])
                    # print(df_det_candidates['iou'].values)
                    # print(f"IoU for current groundtruth object:\n{df_det_candidates['iou']}")

                    # -- Select the best candidate according to IoU if there is one IoU > IOU_THRESHOLD.
                    df_det_candidates = df_det_candidates[df_det_candidates['iou'] > IOU_THRESHOLD]
                    # print(f"Real candidates:\n{df_det_candidates['iou']}")

                    if len(df_det_candidates) != 0:
                        best_iou = df_det_candidates['iou'].max()
                        best_candidate = df_det_candidates[df_det_candidates['iou'] == best_iou]
                        best_candidate = best_candidate.iloc[0]
                        
                        # -- Add the fusion as true positive to the results dataframe.
                        df_results = df_results.append(
                            {
                                'timestamp': gt_row['timestamp'],
                                'type_gt': gt_row['type'],
                                'x_gt': gt_row['x'],
                                'y_gt': gt_row['y'],
                                'z_gt': gt_row['z'],
                                'l_gt': gt_row['l'],
                                'w_gt': gt_row['w'],
                                'h_gt': gt_row['h'],
                                'type_det': best_candidate['type'],
                                'x_det': best_candidate['x'],
                                'y_det': best_candidate['y'],
                                'z_det': best_candidate['z'],
                                'l_det': best_candidate['l'],
                                'w_det': best_candidate['w'],
                                'h_det': best_candidate['h'],
                                'status': 'tp',
                                'iou': df_det_candidates['iou'].values[0],
                            },
                            ignore_index=True,
                        )

                        # -- Mark the groundtruth and the detection as processed in the detection frame.
                        # df_groundtruth.loc[df_groundtruth[df_groundtruth['id'] == gt_index['original_index']].values(0), 'processed'] = True
                        # df_detections.loc[df_det_frame['id'] == best_candidate['original_index'], 'processed'] = True

                        df_gt_frame.loc[gt_index, 'processed'] = True
                        df_det_frame.loc[best_candidate['original_index'], 'processed'] = True

                
                # -- If there are no candidates, mark the groundtruth as false negative.
                elif len(df_det_candidates) == 0 or best_iou < IOU_THRESHOLD:
                    # print(" ************** No candidates found. ************** ")
                    df_results = df_results.append(
                        {
                            'timestamp': gt_row['timestamp'],
                            'type_gt': gt_row['type'],
                            'x_gt': gt_row['x'],
                            'y_gt': gt_row['y'],
                            'z_gt': gt_row['z'],
                            'l_gt': gt_row['l'],
                            'w_gt': gt_row['w'],
                            'h_gt': gt_row['h'],
                            'type_det': None,
                            'x_det': None,
                            'y_det': None,
                            'z_det': None,
                            'l_det': None,
                            'w_det': None,
                            'h_det': None,
                            'status': 'fn',
                            'iou': 0.00,
                        },
                        ignore_index=True,
                    )

                    # -- Mark groundtruth as processed.
                    df_gt_frame.loc[gt_index, 'processed'] = True

                # print(df_detections['processed'])

            # -- Iterate through all detections in current frame.
            for det_index, det_row in df_det_frame.iterrows():

                # -- Filter if they are already processed.
                if det_row['processed'] == False:

                    # -- Add to the results dataframe as false positive.
                    df_results = df_results.append(
                        {
                            'timestamp': det_row['timestamp'],
                            'type_gt': None,
                            'x_gt': None,
                            'y_gt': None,
                            'z_gt': None,
                            'l_gt': None,
                            'w_gt': None,
                            'h_gt': None,
                            'type_det': det_row['type'],
                            'x_det': det_row['x'],
                            'y_det': det_row['y'],
                            'z_det': det_row['z'],
                            'l_det': det_row['l'],
                            'w_det': det_row['w'],
                            'h_det': det_row['h'],
                            'status': 'fp',
                            'iou': 0.00,
                        },
                        ignore_index=True,
                    )

                    # -- Mark the detection as processed.
                    df_det_frame.at[det_index, 'processed'] = True


            # -- Update progress bar
            progress_bar.update(1)

        # -- Close progress bar
        progress_bar.close()

    print("\n\n")
    print("Results:")
    print(tp, fp, fn)

    # -- Save results to CSV
    print(f"Saving results to CSV...")
    df_results.to_csv(f"hola.csv", index=False)
    print(f"Done.")

    # -- Obtain a confusion matrix from the results dataframe.
    df_confusion = df_results.groupby(['status']).count()
    try:
        tp = df_confusion['timestamp'].loc['tp']
    except KeyError:
        tp = 0
    try:
        fp = df_confusion['timestamp'].loc['fp']
    except KeyError:
        fp = 0
    try:
        fn = df_confusion['timestamp'].loc['fn']
    except KeyError:
        fn = 0
    print(df_confusion)

    # -- Calculate precision and recall.
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f05 = 1.25 * (precision * recall) / (0.25 * precision + recall)
    f1 = 2 * (precision * recall) / (precision + recall)
    f2 = 5 * (precision * recall) / (4 * precision + recall)

    # -- Print results.
    print(f"***** Results *****")
    print(f"TP:         {tp}")
    print(f"FP:         {fp}")
    print(f"FN:         {fn}")

    print(f"***** Metrics *****")
    print(f"Precision:  {precision * 100:.2f}%")
    print(f"Recall:     {recall * 100:.2f}%")
    print(f"F0.5:       {f05 * 100:.2f}%")
    print(f"F1:         {f1 * 100:.2f}%")
    print(f"F2:         {f2 * 100:.2f}%")



if __name__ == "__main__":
    main()