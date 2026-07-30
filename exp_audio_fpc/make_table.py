import os, csv

# Make a table of the mean result of each metric
def make_table(results_dir):
    import re
    
    def get_metrics_values(results_dir):
        metrics_values = {}
        
        # Get Smoothness Clap metric value
        # clap_smoothness_clap_csv_path = os.path.join(results_dir, "clap_smoothness_clap_corr_values.csv")
        # with open(clap_smoothness_clap_csv_path, 'r') as f:
        #     reader = list(csv.reader(f))
        #     row = reader[-1] # Get the last row where the mean value is
        #     value_string = row[1]
        #     mean_std_clap = re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?", value_string)
        #     mean_smoothness_clap_corr, std_smoothness_clap_corr = map(float, mean_std_clap)
        #     metrics_values["CLAP Smoothness"] = (mean_smoothness_clap_corr, std_smoothness_clap_corr)

        mert_smoothness_clap_csv_path = os.path.join(results_dir, "MERT_v1-330M", "MERT_v1-330M_smoothness_clap_corr_values.csv")
        with open(mert_smoothness_clap_csv_path, 'r') as f:
            reader = list(csv.reader(f))
            row = reader[-1] # Get the last row where the mean value is
            value_string = row[1]
            mean_std_clap = re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?", value_string)
            mean_smoothness_clap_corr, std_smoothness_clap_corr = map(float, mean_std_clap)
            metrics_values["MERT Smoothness CLAP"] = (mean_smoothness_clap_corr, std_smoothness_clap_corr)
        
        # Get Sobolev k=0, p=2 value
        sobolev_k0_p2_csv_path = os.path.join(results_dir, "MERT_v1-330M", "MERT_v1-330M_sobolev_dists_0_2.csv")
        with open(sobolev_k0_p2_csv_path, 'r') as f:
            reader = list(csv.reader(f))
            row = reader[-1] # Get the last row where the mean value is
            value_string = row[1]
            mean_std_sobolev_k0_p2 = re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?", value_string)
            mean_sobolev_k0_p2, std_sobolev_k0_p2 = map(float, mean_std_sobolev_k0_p2)
            metrics_values["Sobolev (0, 2)"] = (mean_sobolev_k0_p2, std_sobolev_k0_p2)
        
        # Get Sobolev k=1, p=2 value
        sobolev_k1_p2_csv_path = os.path.join(results_dir, "MERT_v1-330M", "MERT_v1-330M_sobolev_dists_1_2.csv")
        with open(sobolev_k1_p2_csv_path, 'r') as f:
            reader = list(csv.reader(f))
            row = reader[-1] # Get the last row where the mean value is
            value_string = row[1]
            mean_std_sobolev_k1_p2 = re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?", value_string)
            mean_sobolev_k1_p2, std_sobolev_k1_p2 = map(float, mean_std_sobolev_k1_p2)
            metrics_values["Sobolev (1, 2)"] = (mean_sobolev_k1_p2, std_sobolev_k1_p2)
        
        # Get Correspondence value
        # mfcc_correspondence_csv_path = os.path.join(results_dir, "mfcc_soundmorpher_correspondence_mfccs_values.csv")
        # with open(mfcc_correspondence_csv_path, 'r') as f:
        #     reader = list(csv.reader(f))
        #     row = reader[-1] # Get the last row where the mean value is
        #     value_string = row[1]
        #     mean_std_correspondence = re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?", value_string)
        #     mean_correspondence, std_correspondence = map(float, mean_std_correspondence)
        #     metrics_values["MFCC Correspondence"] = (mean_correspondence, std_correspondence)

        mert_correspondence_csv_path = os.path.join(results_dir, "MERT_v1-330M", "MERT_v1-330M_correspondence_mfccs_values.csv")
        with open(mert_correspondence_csv_path, 'r') as f:
            reader = list(csv.reader(f))
            row = reader[-1] # Get the last row where the mean value is
            value_string = row[1]
            mean_std_correspondence = re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?", value_string)
            mean_correspondence, std_correspondence = map(float, mean_std_correspondence)
            metrics_values["MERT Correspondence"] = (mean_correspondence, std_correspondence)
        
        # Get Intermediateness value
        # cdpam_intermediateness_csv_path = os.path.join(results_dir, "cdpam_intermediateness_total_cdpam_values.csv")
        # with open(cdpam_intermediateness_csv_path, 'r') as f:
        #     reader = list(csv.reader(f))
        #     row = reader[-1] # Get the last row where the mean value is
        #     value_string = row[1]
        #     mean_std_intermediateness = re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?", value_string)
        #     mean_intermediateness, std_intermediateness = map(float, mean_std_intermediateness)
        #     metrics_values["CDPAM Intermediateness"] = (mean_intermediateness, std_intermediateness)

        mert_intermediateness_csv_path = os.path.join(results_dir, "MERT_v1-330M", "MERT_v1-330M_intermediateness_total_cdpam_values.csv")
        with open(mert_intermediateness_csv_path, 'r') as f:
            reader = list(csv.reader(f))
            row = reader[-1] # Get the last row where the mean value is
            value_string = row[1]
            mean_std_intermediateness = re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?", value_string)
            mean_intermediateness, std_intermediateness = map(float, mean_std_intermediateness)
            metrics_values["MERT Intermediateness"] = (mean_intermediateness, std_intermediateness)

        # Get Smoothness CDPAM value
        # cdpam_smoothness_cdpam_csv_path = os.path.join(results_dir, "cdpam_smoothness_mean_cdpam_values.csv")
        # with open(cdpam_smoothness_cdpam_csv_path, 'r') as f:
        #     reader = list(csv.reader(f))
        #     row = reader[-1] # Get the last row where the mean value is
        #     value_string = row[1]
        #     mean_std_smoothness_cdpam = re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?", value_string)
        #     mean_smoothness_cdpam, std_smoothness_cdpam = map(float, mean_std_smoothness_cdpam)
        #     metrics_values["CDPAM Smoothness CDPAM"] = (mean_smoothness_cdpam, std_smoothness_cdpam)

        mert_smoothness_cdpam_csv_path = os.path.join(results_dir, "MERT_v1-330M", "MERT_v1-330M_smoothness_mean_cdpam_values.csv")
        with open(mert_smoothness_cdpam_csv_path, 'r') as f:
            reader = list(csv.reader(f))
            row = reader[-1] # Get the last row where the mean value is
            value_string = row[1]
            mean_std_smoothness_cdpam = re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?", value_string)
            mean_smoothness_cdpam, std_smoothness_cdpam = map(float, mean_std_smoothness_cdpam)
            metrics_values["MERT Smoothness CDPAM"] = (mean_smoothness_cdpam, std_smoothness_cdpam)
        
        return metrics_values
        
    # results_dir_fpc1 = "exp_invalidate_metrics_fpc/generated/metrics/fpc1"
    # fpc1_metrics_values = get_metrics_values(results_dir_fpc1)
    # results_dir_fpc2 = "exp_invalidate_metrics_fpc/generated/metrics/fpc2"
    # fpc2_metrics_values = get_metrics_values(results_dir_fpc2)
    # results_dir_fpc3 = "exp_invalidate_metrics_fpc/generated/metrics/fpc3"
    # fpc3_metrics_values = get_metrics_values(results_dir_fpc3)
    results_dir_linear = f"{results_dir}/experiment"
    linear_metrics_values = get_metrics_values(results_dir_linear)
    results_dir_random = f"{results_dir}/random"
    random_metrics_values = get_metrics_values(results_dir_random)

    # Write the table to a CSV file
    output_csv_path = os.path.join(results_dir, "metrics_table.csv")
    with open(output_csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # Write header: metrics as rows
        header = ["Metric", "Encoder", "Ideal Value", "Random Value"]#, "FPC 1 Value", "FPC 2 Value", "FPC 3 Value"]
        writer.writerow(header)

        # Write rows: models as rows, (k, p) as columns, mean+-std as values
        # row = [
        #     "Correspondence", 
        #     "MFCC",
        #     f"{linear_metrics_values['MFCC Correspondence'][0]:.2f} +- {linear_metrics_values['MFCC Correspondence'][1]:.2f}",
        #     f"{random_metrics_values['MFCC Correspondence'][0]:.2f} +- {random_metrics_values['MFCC Correspondence'][1]:.2f}",
        #     # f"{fpc1_metrics_values['MFCC Correspondence'][0]:.2f} +- {fpc1_metrics_values['MFCC Correspondence'][1]:.2f}",
        #     # f"{fpc2_metrics_values['MFCC Correspondence'][0]:.2f} +- {fpc2_metrics_values['MFCC Correspondence'][1]:.2f}",
        #     # f"{fpc3_metrics_values['MFCC Correspondence'][0]:.2f} +- {fpc3_metrics_values['MFCC Correspondence'][1]:.2f}"
        # ]
        # writer.writerow(row)
        # row = [
        #     "Smoothness CLAP",
        #     "L-CLAP audio",
        #     f"{linear_metrics_values['CLAP Smoothness'][0]:.2f} +- {linear_metrics_values['CLAP Smoothness'][1]:.2f}",
        #     f"{random_metrics_values['CLAP Smoothness'][0]:.2f} +- {random_metrics_values['CLAP Smoothness'][1]:.2f}",
        #     # f"{fpc1_metrics_values['CLAP Smoothness'][0]:.2f} +- {fpc1_metrics_values['CLAP Smoothness'][1]:.2f}",
        #     # f"{fpc2_metrics_values['CLAP Smoothness'][0]:.2f} +- {fpc2_metrics_values['CLAP Smoothness'][1]:.2f}",
        #     # f"{fpc3_metrics_values['CLAP Smoothness'][0]:.2f} +- {fpc3_metrics_values['CLAP Smoothness'][1]:.2f}"
        # ]
        # writer.writerow(row)
        # row = [
        #     "Intermediateness",
        #     "CDPAM",
        #     f"{linear_metrics_values['CDPAM Intermediateness'][0]:.2f} +- {linear_metrics_values['CDPAM Intermediateness'][1]:.2f}",
        #     f"{random_metrics_values['CDPAM Intermediateness'][0]:.2f} +- {random_metrics_values['CDPAM Intermediateness'][1]:.2f}",
        #     # f"{fpc1_metrics_values['CDPAM Intermediateness'][0]:.2f} +- {fpc1_metrics_values['CDPAM Intermediateness'][1]:.2f}",
        #     # f"{fpc2_metrics_values['CDPAM Intermediateness'][0]:.2f} +- {fpc2_metrics_values['CDPAM Intermediateness'][1]:.2f}",
        #     # f"{fpc3_metrics_values['CDPAM Intermediateness'][0]:.2f} +- {fpc3_metrics_values['CDPAM Intermediateness'][1]:.2f}"
        # ]
        # writer.writerow(row)
        # row = [
        #     "Smoothness CDPAM",
        #     "CDPAM",
        #     f"{linear_metrics_values['CDPAM Smoothness CDPAM'][0]:.2f} +- {linear_metrics_values['CDPAM Smoothness CDPAM'][1]:.2f}",
        #     f"{random_metrics_values['CDPAM Smoothness CDPAM'][0]:.2f} +- {random_metrics_values['CDPAM Smoothness CDPAM'][1]:.2f}",
        #     # f"{fpc1_metrics_values['CDPAM Smoothness CDPAM'][0]:.2f} +- {fpc1_metrics_values['CDPAM Smoothness CDPAM'][1]:.2f}",
        #     # f"{fpc2_metrics_values['CDPAM Smoothness CDPAM'][0]:.2f} +- {fpc2_metrics_values['CDPAM Smoothness CDPAM'][1]:.2f}",
        #     # f"{fpc3_metrics_values['CDPAM Smoothness CDPAM'][0]:.2f} +- {fpc3_metrics_values['CDPAM Smoothness CDPAM'][1]:.2f}"
        # ]
        # writer.writerow(row)

        # MERT
        row = [
            "Correspondence", 
            "MERT",
            f"{linear_metrics_values['MERT Correspondence'][0]:.2f} +- {linear_metrics_values['MERT Correspondence'][1]:.2f}",
            f"{random_metrics_values['MERT Correspondence'][0]:.2f} +- {random_metrics_values['MERT Correspondence'][1]:.2f}",
            # f"{fpc1_metrics_values['MERT Correspondence'][0]:.2f} +- {fpc1_metrics_values['MERT Correspondence'][1]:.2f}",
            # f"{fpc2_metrics_values['MERT Correspondence'][0]:.2f} +- {fpc2_metrics_values['MERT Correspondence'][1]:.2f}",
            # f"{fpc3_metrics_values['MERT Correspondence'][0]:.2f} +- {fpc3_metrics_values['MERT Correspondence'][1]:.2f}"
        ]
        writer.writerow(row)
        row = [
            "Smoothness CLAP",
            "MERT",
            f"{linear_metrics_values['MERT Smoothness CLAP'][0]:.2f} +- {linear_metrics_values['MERT Smoothness CLAP'][1]:.2f}",
            f"{random_metrics_values['MERT Smoothness CLAP'][0]:.2f} +- {random_metrics_values['MERT Smoothness CLAP'][1]:.2f}",
            # f"{fpc1_metrics_values['MERT Smoothness CLAP'][0]:.2f} +- {fpc1_metrics_values['MERT Smoothness CLAP'][1]:.2f}",
            # f"{fpc2_metrics_values['MERT Smoothness CLAP'][0]:.2f} +- {fpc2_metrics_values['MERT Smoothness CLAP'][1]:.2f}",
            # f"{fpc3_metrics_values['MERT Smoothness CLAP'][0]:.2f} +- {fpc3_metrics_values['MERT Smoothness CLAP'][1]:.2f}"
        ]
        writer.writerow(row)
        row = [
            "Intermediateness",
            "MERT",
            f"{linear_metrics_values['MERT Intermediateness'][0]:.2f} +- {linear_metrics_values['MERT Intermediateness'][1]:.2f}",
            f"{random_metrics_values['MERT Intermediateness'][0]:.2f} +- {random_metrics_values['MERT Intermediateness'][1]:.2f}",
            # f"{fpc1_metrics_values['MERT Intermediateness'][0]:.2f} +- {fpc1_metrics_values['MERT Intermediateness'][1]:.2f}",
            # f"{fpc2_metrics_values['MERT Intermediateness'][0]:.2f} +- {fpc2_metrics_values['MERT Intermediateness'][1]:.2f}",
            # f"{fpc3_metrics_values['MERT Intermediateness'][0]:.2f} +- {fpc3_metrics_values['MERT Intermediateness'][1]:.2f}"
        ]
        writer.writerow(row)
        row = [
            "Smoothness CDPAM",
            "MERT",
            f"{linear_metrics_values['MERT Smoothness CDPAM'][0]:.2f} +- {linear_metrics_values['MERT Smoothness CDPAM'][1]:.2f}",
            f"{random_metrics_values['MERT Smoothness CDPAM'][0]:.2f} +- {random_metrics_values['MERT Smoothness CDPAM'][1]:.2f}",
            # f"{fpc1_metrics_values['MERT Smoothness CDPAM'][0]:.2f} +- {fpc1_metrics_values['MERT Smoothness CDPAM'][1]:.2f}",
            # f"{fpc2_metrics_values['MERT Smoothness CDPAM'][0]:.2f} +- {fpc2_metrics_values['MERT Smoothness CDPAM'][1]:.2f}",
            # f"{fpc3_metrics_values['MERT Smoothness CDPAM'][0]:.2f} +- {fpc3_metrics_values['MERT Smoothness CDPAM'][1]:.2f}"
        ]
        writer.writerow(row)
        row = [
            "Sobolev (0, 2)",
            "MERT",
            f"{linear_metrics_values['Sobolev (0, 2)'][0]:.2f} +- {linear_metrics_values['Sobolev (0, 2)'][1]:.2f}",
            f"{random_metrics_values['Sobolev (0, 2)'][0]:.2f} +- {random_metrics_values['Sobolev (0, 2)'][1]:.2f}",
            # f"{fpc1_metrics_values['Sobolev (0, 2)'][0]:.2f} +- {fpc1_metrics_values['Sobolev (0, 2)'][1]:.2f}",
            # f"{fpc2_metrics_values['Sobolev (0, 2)'][0]:.2f} +- {fpc2_metrics_values['Sobolev (0, 2)'][1]:.2f}",
            # f"{fpc3_metrics_values['Sobolev (0, 2)'][0]:.2f} +- {fpc3_metrics_values['Sobolev (0, 2)'][1]:.2f}"
        ]
        writer.writerow(row)
        row = [
            "Sobolev (1, 2)",
            "MERT",
            f"{linear_metrics_values['Sobolev (1, 2)'][0]:.2f} +- {linear_metrics_values['Sobolev (1, 2)'][1]:.2f}",
            f"{random_metrics_values['Sobolev (1, 2)'][0]:.2f} +- {random_metrics_values['Sobolev (1, 2)'][1]:.2f}",
            # f"{fpc1_metrics_values['Sobolev (1, 2)'][0]:.2f} +- {fpc1_metrics_values['Sobolev (1, 2)'][1]:.2f}",
            # f"{fpc2_metrics_values['Sobolev (1, 2)'][0]:.2f} +- {fpc2_metrics_values['Sobolev (1, 2)'][1]:.2f}",
            # f"{fpc3_metrics_values['Sobolev (1, 2)'][0]:.2f} +- {fpc3_metrics_values['Sobolev (1, 2)'][1]:.2f}"
        ]
        writer.writerow(row)
    