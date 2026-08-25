import os, csv

# Make a table of the mean result of each metric
def make_table(results_dir):
    import re
    
    def get_metrics_values(results_dir, no_audio: bool = False):
        metrics_values = {}
        
        # Get Smoothness Clap metric value
        clap_smoothness_clap_csv_path = os.path.join(results_dir, "LaionCLAP_audio", "LaionCLAP_audio_smoothness_clap_corr_values.csv")
        with open(clap_smoothness_clap_csv_path, 'r') as f:
            reader = list(csv.reader(f))
            row = reader[-1] # Get the last row where the mean value is
            value_string = row[1]
            mean_std_clap = re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?", value_string)
            mean_smoothness_clap_corr, std_smoothness_clap_corr = map(float, mean_std_clap)
            metrics_values["CLAP Smoothness"] = (mean_smoothness_clap_corr, std_smoothness_clap_corr)

        mert_smoothness_clap_csv_path = os.path.join(results_dir, "MERT_v1-330M", "MERT_v1-330M_smoothness_clap_corr_values.csv")
        with open(mert_smoothness_clap_csv_path, 'r') as f:
            reader = list(csv.reader(f))
            row = reader[-1] # Get the last row where the mean value is
            value_string = row[1]
            mean_std_clap = re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?", value_string)
            mean_smoothness_clap_corr, std_smoothness_clap_corr = map(float, mean_std_clap)
            metrics_values["MERT Smoothness MF"] = (mean_smoothness_clap_corr, std_smoothness_clap_corr)
        
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
        
        # Get Correspondence SM value
        if no_audio:
            metrics_values["MFCC Correspondence SM"] = (404.0, 404.0)
        else:
            mfcc_correspondence_csv_path = os.path.join(results_dir, "MFCC", "MFCC_correspondence_mfccs_values.csv")
            with open(mfcc_correspondence_csv_path, 'r') as f:
                reader = list(csv.reader(f))
                row = reader[-1] # Get the last row where the mean value is
                value_string = row[1]
                mean_std_correspondence = re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?", value_string)
                mean_correspondence, std_correspondence = map(float, mean_std_correspondence)
                metrics_values["MFCC Correspondence SM"] = (mean_correspondence, std_correspondence)

        mert_correspondence_csv_path = os.path.join(results_dir, "MERT_v1-330M", "MERT_v1-330M_correspondence_mfccs_values.csv")
        with open(mert_correspondence_csv_path, 'r') as f:
            reader = list(csv.reader(f))
            row = reader[-1] # Get the last row where the mean value is
            value_string = row[1]
            mean_std_correspondence = re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?", value_string)
            mean_correspondence, std_correspondence = map(float, mean_std_correspondence)
            metrics_values["MERT Correspondence SM"] = (mean_correspondence, std_correspondence)
        
        # Get Intermediateness SM value
        if no_audio:
            metrics_values["CDPAM Intermediateness SM"] = (404.0, 404.0)
        else:
            cdpam_intermediateness_csv_path = os.path.join(results_dir, "CDPAM", "CDPAM_intermediateness_total_cdpam_values.csv")
            with open(cdpam_intermediateness_csv_path, 'r') as f:
                reader = list(csv.reader(f))
                row = reader[-1] # Get the last row where the mean value is
                value_string = row[1]
                mean_std_intermediateness = re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?", value_string)
                mean_intermediateness, std_intermediateness = map(float, mean_std_intermediateness)
                metrics_values["CDPAM Intermediateness SM"] = (mean_intermediateness, std_intermediateness)

        mert_intermediateness_csv_path = os.path.join(results_dir, "MERT_v1-330M", "MERT_v1-330M_intermediateness_total_cdpam_values.csv")
        with open(mert_intermediateness_csv_path, 'r') as f:
            reader = list(csv.reader(f))
            row = reader[-1] # Get the last row where the mean value is
            value_string = row[1]
            mean_std_intermediateness = re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?", value_string)
            mean_intermediateness, std_intermediateness = map(float, mean_std_intermediateness)
            metrics_values["MERT Intermediateness SM"] = (mean_intermediateness, std_intermediateness)

        # Get Smoothness SM value
        if no_audio:
            metrics_values["CDPAM Smoothness SM"] = (404.0, 404.0)
        else:
            cdpam_smoothness_cdpam_csv_path = os.path.join(results_dir, "CDPAM", "CDPAM_smoothness_mean_cdpam_values.csv")
            with open(cdpam_smoothness_cdpam_csv_path, 'r') as f:
                reader = list(csv.reader(f))
                row = reader[-1] # Get the last row where the mean value is
                value_string = row[1]
                mean_std_smoothness_cdpam = re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?", value_string)
                mean_smoothness_cdpam, std_smoothness_cdpam = map(float, mean_std_smoothness_cdpam)
                metrics_values["CDPAM Smoothness SM"] = (mean_smoothness_cdpam, std_smoothness_cdpam)

        mert_smoothness_cdpam_csv_path = os.path.join(results_dir, "MERT_v1-330M", "MERT_v1-330M_smoothness_mean_cdpam_values.csv")
        with open(mert_smoothness_cdpam_csv_path, 'r') as f:
            reader = list(csv.reader(f))
            row = reader[-1] # Get the last row where the mean value is
            value_string = row[1]
            mean_std_smoothness_cdpam = re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?", value_string)
            mean_smoothness_cdpam, std_smoothness_cdpam = map(float, mean_std_smoothness_cdpam)
            metrics_values["MERT Smoothness SM"] = (mean_smoothness_cdpam, std_smoothness_cdpam)
        
        return metrics_values
    
    results_dir_linear = f"{results_dir}/experiment"
    linear_metrics_values = get_metrics_values(results_dir_linear)
    results_dir_random = f"{results_dir}/random"
    random_metrics_values = get_metrics_values(results_dir_random)
        
    results_dir_nuc = f"{results_dir}/nuc"
    nuc_metrics_values = get_metrics_values(results_dir_nuc)

    results_dir_eqc = f"{results_dir}/normalized_eqc" # f"{results_dir}/eqc"
    eqc_metrics_values = get_metrics_values(results_dir_eqc)
    results_dir_emb_eqc = f"{results_dir}/emb_eqc" # f"{results_dir}/normalized_eqc" # f"{results_dir}/eqc"
    emb_eqc_metrics_values = get_metrics_values(results_dir_emb_eqc, no_audio=True)
    results_dir_emb_nuc = f"{results_dir}/emb_nuc" # f"{results_dir}/normalized_nuc" # f"{results_dir}/nuc"
    emb_nuc_metrics_values = get_metrics_values(results_dir_emb_nuc, no_audio=True)

    # Write the table to a CSV file
    output_csv_path = os.path.join(results_dir, "metrics_table.csv")
    with open(output_csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # Write header: metrics as rows
        header = ["Metric", "Encoder", "Morph", "Null", "NUC", "Emb NUC", "EQC", "Emb EQC"]
        writer.writerow(header)

        # Write rows: models as rows, (k, p) as columns, mean+-std as values
        row = [
            "Correspondence SM", 
            "MFCC",
            f"{linear_metrics_values['MFCC Correspondence SM'][0]:.2f} +- {linear_metrics_values['MFCC Correspondence SM'][1]:.2f}",
            f"{random_metrics_values['MFCC Correspondence SM'][0]:.2f} +- {random_metrics_values['MFCC Correspondence SM'][1]:.2f}",
            f"{nuc_metrics_values['MFCC Correspondence SM'][0]:.2f} +- {nuc_metrics_values['MFCC Correspondence SM'][1]:.2f}",
            f"{emb_nuc_metrics_values['MFCC Correspondence SM'][0]:.2f} +- {emb_nuc_metrics_values['MFCC Correspondence SM'][1]:.2f}",
            f"{eqc_metrics_values['MFCC Correspondence SM'][0]:.2f} +- {eqc_metrics_values['MFCC Correspondence SM'][1]:.2f}",
            f"{emb_eqc_metrics_values['MFCC Correspondence SM'][0]:.2f} +- {emb_eqc_metrics_values['MFCC Correspondence SM'][1]:.2f}",
        ]
        writer.writerow(row)
        row = [
            "Smoothness MF",
            "L-CLAP audio",
            f"{linear_metrics_values['CLAP Smoothness'][0]:.2f} +- {linear_metrics_values['CLAP Smoothness'][1]:.2f}",
            f"{random_metrics_values['CLAP Smoothness'][0]:.2f} +- {random_metrics_values['CLAP Smoothness'][1]:.2f}",
            f"{nuc_metrics_values['CLAP Smoothness'][0]:.2f} +- {nuc_metrics_values['CLAP Smoothness'][1]:.2f}",
            f"{emb_nuc_metrics_values['CLAP Smoothness'][0]:.2f} +- {emb_nuc_metrics_values['CLAP Smoothness'][1]:.2f}",
            f"{eqc_metrics_values['CLAP Smoothness'][0]:.2f} +- {eqc_metrics_values['CLAP Smoothness'][1]:.2f}",
            f"{emb_eqc_metrics_values['CLAP Smoothness'][0]:.2f} +- {emb_eqc_metrics_values['CLAP Smoothness'][1]:.2f}"
        ]
        writer.writerow(row)
        row = [
            "Intermediateness SM",
            "CDPAM",
            f"{linear_metrics_values['CDPAM Intermediateness SM'][0]:.2f} +- {linear_metrics_values['CDPAM Intermediateness SM'][1]:.2f}",
            f"{random_metrics_values['CDPAM Intermediateness SM'][0]:.2f} +- {random_metrics_values['CDPAM Intermediateness SM'][1]:.2f}",
            f"{nuc_metrics_values['CDPAM Intermediateness SM'][0]:.2f} +- {nuc_metrics_values['CDPAM Intermediateness SM'][1]:.2f}",
            f"{emb_nuc_metrics_values['CDPAM Intermediateness SM'][0]:.2f} +- {emb_nuc_metrics_values['CDPAM Intermediateness SM'][1]:.2f}",
            f"{eqc_metrics_values['CDPAM Intermediateness SM'][0]:.2f} +- {eqc_metrics_values['CDPAM Intermediateness SM'][1]:.2f}",
            f"{emb_eqc_metrics_values['CDPAM Intermediateness SM'][0]:.2f} +- {emb_eqc_metrics_values['CDPAM Intermediateness SM'][1]:.2f}"
        ]
        writer.writerow(row)
        row = [
            "Smoothness SM",
            "CDPAM",
            f"{linear_metrics_values['CDPAM Smoothness SM'][0]:.2f} +- {linear_metrics_values['CDPAM Smoothness SM'][1]:.2f}",
            f"{random_metrics_values['CDPAM Smoothness SM'][0]:.2f} +- {random_metrics_values['CDPAM Smoothness SM'][1]:.2f}",
            f"{nuc_metrics_values['CDPAM Smoothness SM'][0]:.2f} +- {nuc_metrics_values['CDPAM Smoothness SM'][1]:.2f}",
            f"{emb_nuc_metrics_values['CDPAM Smoothness SM'][0]:.2f} +- {emb_nuc_metrics_values['CDPAM Smoothness SM'][1]:.2f}",
            f"{eqc_metrics_values['CDPAM Smoothness SM'][0]:.2f} +- {eqc_metrics_values['CDPAM Smoothness SM'][1]:.2f}",
            f"{emb_eqc_metrics_values['CDPAM Smoothness SM'][0]:.2f} +- {emb_eqc_metrics_values['CDPAM Smoothness SM'][1]:.2f}"
        ]
        writer.writerow(row)

        # MERT
        row = [
            "Correspondence SM", 
            "MERT",
            f"{linear_metrics_values['MERT Correspondence SM'][0]:.2f} +- {linear_metrics_values['MERT Correspondence SM'][1]:.2f}",
            f"{random_metrics_values['MERT Correspondence SM'][0]:.2f} +- {random_metrics_values['MERT Correspondence SM'][1]:.2f}",
            f"{nuc_metrics_values['MERT Correspondence SM'][0]:.2f} +- {nuc_metrics_values['MERT Correspondence SM'][1]:.2f}",
            f"{emb_nuc_metrics_values['MERT Correspondence SM'][0]:.2f} +- {emb_nuc_metrics_values['MERT Correspondence SM'][1]:.2f}",
            f"{eqc_metrics_values['MERT Correspondence SM'][0]:.2f} +- {eqc_metrics_values['MERT Correspondence SM'][1]:.2f}",
            f"{emb_eqc_metrics_values['MERT Correspondence SM'][0]:.2f} +- {emb_eqc_metrics_values['MERT Correspondence SM'][1]:.2f}"
        ]
        writer.writerow(row)
        row = [
            "Smoothness MF",
            "MERT",
            f"{linear_metrics_values['MERT Smoothness MF'][0]:.2f} +- {linear_metrics_values['MERT Smoothness MF'][1]:.2f}",
            f"{random_metrics_values['MERT Smoothness MF'][0]:.2f} +- {random_metrics_values['MERT Smoothness MF'][1]:.2f}",
            f"{nuc_metrics_values['MERT Smoothness MF'][0]:.2f} +- {nuc_metrics_values['MERT Smoothness MF'][1]:.2f}",
            f"{emb_nuc_metrics_values['MERT Smoothness MF'][0]:.2f} +- {emb_nuc_metrics_values['MERT Smoothness MF'][1]:.2f}",
            f"{eqc_metrics_values['MERT Smoothness MF'][0]:.2f} +- {eqc_metrics_values['MERT Smoothness MF'][1]:.2f}",
            f"{emb_eqc_metrics_values['MERT Smoothness MF'][0]:.2f} +- {emb_eqc_metrics_values['MERT Smoothness MF'][1]:.2f}"
        ]
        writer.writerow(row)
        row = [
            "Intermediateness SM",
            "MERT",
            f"{linear_metrics_values['MERT Intermediateness SM'][0]:.2f} +- {linear_metrics_values['MERT Intermediateness SM'][1]:.2f}",
            f"{random_metrics_values['MERT Intermediateness SM'][0]:.2f} +- {random_metrics_values['MERT Intermediateness SM'][1]:.2f}",
            f"{nuc_metrics_values['MERT Intermediateness SM'][0]:.2f} +- {nuc_metrics_values['MERT Intermediateness SM'][1]:.2f}",
            f"{emb_nuc_metrics_values['MERT Intermediateness SM'][0]:.2f} +- {emb_nuc_metrics_values['MERT Intermediateness SM'][1]:.2f}",
            f"{eqc_metrics_values['MERT Intermediateness SM'][0]:.2f} +- {eqc_metrics_values['MERT Intermediateness SM'][1]:.2f}",
            f"{emb_eqc_metrics_values['MERT Intermediateness SM'][0]:.2f} +- {emb_eqc_metrics_values['MERT Intermediateness SM'][1]:.2f}"
        ]
        writer.writerow(row)
        row = [
            "Smoothness SM",
            "MERT",
            f"{linear_metrics_values['MERT Smoothness SM'][0]:.2f} +- {linear_metrics_values['MERT Smoothness SM'][1]:.2f}",
            f"{random_metrics_values['MERT Smoothness SM'][0]:.2f} +- {random_metrics_values['MERT Smoothness SM'][1]:.2f}",
            f"{nuc_metrics_values['MERT Smoothness SM'][0]:.2f} +- {nuc_metrics_values['MERT Smoothness SM'][1]:.2f}",
            f"{emb_nuc_metrics_values['MERT Smoothness SM'][0]:.2f} +- {emb_nuc_metrics_values['MERT Smoothness SM'][1]:.2f}",
            f"{eqc_metrics_values['MERT Smoothness SM'][0]:.2f} +- {eqc_metrics_values['MERT Smoothness SM'][1]:.2f}",
            f"{emb_eqc_metrics_values['MERT Smoothness SM'][0]:.2f} +- {emb_eqc_metrics_values['MERT Smoothness SM'][1]:.2f}"
        ]
        writer.writerow(row)
        row = [
            "Sobolev (0, 2)",
            "MERT",
            f"{linear_metrics_values['Sobolev (0, 2)'][0]:.2f} +- {linear_metrics_values['Sobolev (0, 2)'][1]:.2f}",
            f"{random_metrics_values['Sobolev (0, 2)'][0]:.2f} +- {random_metrics_values['Sobolev (0, 2)'][1]:.2f}",
            f"{nuc_metrics_values['Sobolev (0, 2)'][0]:.2f} +- {nuc_metrics_values['Sobolev (0, 2)'][1]:.2f}",
            f"{emb_nuc_metrics_values['Sobolev (0, 2)'][0]:.2f} +- {emb_nuc_metrics_values['Sobolev (0, 2)'][1]:.2f}",
            f"{eqc_metrics_values['Sobolev (0, 2)'][0]:.2f} +- {eqc_metrics_values['Sobolev (0, 2)'][1]:.2f}",
            f"{emb_eqc_metrics_values['Sobolev (0, 2)'][0]:.2f} +- {emb_eqc_metrics_values['Sobolev (0, 2)'][1]:.2f}"
        ]
        writer.writerow(row)
        row = [
            "Sobolev (1, 2)",
            "MERT",
            f"{linear_metrics_values['Sobolev (1, 2)'][0]:.2f} +- {linear_metrics_values['Sobolev (1, 2)'][1]:.2f}",
            f"{random_metrics_values['Sobolev (1, 2)'][0]:.2f} +- {random_metrics_values['Sobolev (1, 2)'][1]:.2f}",
            f"{nuc_metrics_values['Sobolev (1, 2)'][0]:.2f} +- {nuc_metrics_values['Sobolev (1, 2)'][1]:.2f}",
            f"{emb_nuc_metrics_values['Sobolev (1, 2)'][0]:.2f} +- {emb_nuc_metrics_values['Sobolev (1, 2)'][1]:.2f}",
            f"{eqc_metrics_values['Sobolev (1, 2)'][0]:.2f} +- {eqc_metrics_values['Sobolev (1, 2)'][1]:.2f}",
            f"{emb_eqc_metrics_values['Sobolev (1, 2)'][0]:.2f} +- {emb_eqc_metrics_values['Sobolev (1, 2)'][1]:.2f}"
        ]
        writer.writerow(row)

    print(f"Table generated successfully at {output_csv_path}.")

make_table("exp_audio_fpc/generated/results")