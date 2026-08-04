import shutil
import os
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from os.path import join, exists

from deep_utils import StringUtils, JsonUtils

from sklearn.model_selection import train_test_split

parser = ArgumentParser(
    description="Single entry point for the CCT-FM benchmarking: preprocessing, training and "
                "testing of the nnUZoo models. Exports the nnUNet_raw / nnUNet_preprocessed / "
                "nnUNet_results environment variables, then shells out to "
                "nnUNetv2_plan_and_preprocess, nnUNetv2_train, code_inference.py and "
                "nnUNetv2_evaluate_folder. Pick one or more stages with --preprocess / --train / "
                "--test; without a stage flag nothing is run.",
    epilog="Example:\n"
           "  python train.py -dataset_name Dataset051_CT_Cardio_FULL_Organs -device 0 \\\n"
           "      -tr nnUNetTrainerMetalArtifactStepMotionNoMirror --train --c\n"
           "\n"
           "See README.md for the trainer name of every benchmarked model and for the external\n"
           "test-set folder names.",
    formatter_class=RawDescriptionHelpFormatter,
)

stage = parser.add_argument_group("stages", "which steps to run; combinable in a single call")
stage.add_argument("--preprocess", action="store_true",
                   help="Plan and preprocess the dataset. Run once per dataset -- the result is "
                        "shared by every trainer and every architecture.")
stage.add_argument("--train", action="store_true",
                   help="Train the model given by -tr on -fold of -dataset_name.")
stage.add_argument("--test", action="store_true",
                   help="Predict on the test set, reverse the label mapping if dataset.json "
                        "declares 'reverse_class', then compute the metrics into "
                        "-test_summary_name.")

data = parser.add_argument_group("dataset and configuration")
data.add_argument("-dataset_name", required=True,
                  help="nnU-Net dataset folder name, e.g. Dataset051_CT_Cardio_FULL_Organs. The "
                       "3-digit dataset id needed for preprocessing is read out of it.")
data.add_argument("-fold", default=0, type=int,
                  help="Cross-validation fold. The study reports fold 0 throughout. (default: 0)")
data.add_argument("-model", default="3d_fullres",
                  help="nnU-Net configuration: 3d_fullres, 3d_lowres, 2d, ... "
                       "(default: 3d_fullres)")
data.add_argument("-tr", default="nnUNetTrainer",
                  help="Trainer class name; this is what selects the architecture and the "
                       "augmentation pipeline. nnUNetTrainer is plain nnU-Net without CT-Aug; see "
                       "the model table in README.md for the rest. (default: nnUNetTrainer)")
data.add_argument("-p", default="nnUNetResEncUNetLPlans",
                  help="Plans identifier used for training and testing. All benchmarks use the "
                       "residual-encoder L preset. (default: nnUNetResEncUNetLPlans)")
data.add_argument("-pl", default="nnUNetPlannerResEncL",
                  help="Experiment planner used by --preprocess; must be the planner that writes "
                       "the -p plans. (default: nnUNetPlannerResEncL)")

paths = parser.add_argument_group(
    "paths",
    "Either pass -base_dir alone, or all three of -nnunet_raw / -nnunet_preprocessed / "
    "-nnunet_results. Mixing the two forms is rejected; passing neither falls back to a "
    "per-hostname default.")
paths.add_argument("-base_dir", default=None, type=str,
                   help="Root holding nnunet_raw/, nnunet_preprocessed/ and nnunet_results/.")
paths.add_argument("-nnunet_raw", default=None, type=str,
                   help="Raw dataset root (nnUNet_raw).")
paths.add_argument("-nnunet_preprocessed", default=None, type=str,
                   help="Preprocessed data root (nnUNet_preprocessed).")
paths.add_argument("-nnunet_results", default=None, type=str,
                   help="Results/checkpoints root (nnUNet_results).")
paths.add_argument("-bin_path", default="", type=str,
                   help="bin/ directory of the environment providing the nnUNetv2_* console "
                        "scripts. Needed only when driving an environment other than the active "
                        "one, e.g. launching the Mamba environment from the main one. Empty means "
                        "'take them from PATH'.")

train_g = parser.add_argument_group("training (--train)")
train_g.add_argument("-device", type=int, default=[0], nargs="+",
                     help="GPU index, or several for multi-GPU. Exported as CUDA_VISIBLE_DEVICES, "
                          "so the trainer itself always sees cuda:0. (default: 0)")
train_g.add_argument("-num_gpus", default=1, type=int,
                     help="Number of GPUs to train on; must match the number of -device indices. "
                          "(default: 1)")
train_g.add_argument("-num_epochs", type=int, default=1000,
                     help="Training epochs. The benchmarked models use 1000; the fine-tuning runs "
                          "use far fewer. (default: 1000)")
train_g.add_argument("-batch_size", type=int, default=None,
                     help="Override the batch size chosen by the plans. Use this to fit a large "
                          "patch size on a smaller GPU.")
train_g.add_argument("-pretrained_weights", type=str, default=None,
                     help="Checkpoint to initialise from, e.g. the self-supervised foundation "
                          "model. Resolved relative to nnUNet_results when the path does not "
                          "exist as given. Ignored when --c resumes an existing run.")
train_g.add_argument("--only_encoder", action="store_true",
                     help="Load only the encoder from -pretrained_weights and leave the decoder "
                          "randomly initialised.")
train_g.add_argument("--c", action="store_true",
                     help="Resume from the latest checkpoint, and skip the run entirely when its "
                          "validation summary already exists -- safe to re-issue over a whole "
                          "sweep. Falls back to a fresh run if no checkpoint is there yet. "
                          "Mutually exclusive with --overwrite_train.")
train_g.add_argument("--overwrite_train", action="store_true",
                     help="Retrain from scratch even if a checkpoint exists. Cannot be combined "
                          "with --c.")
train_g.add_argument("--val", action="store_true",
                     help="Skip training and only run the validation pass of an existing run.")
train_g.add_argument("--val_best", action="store_true",
                     help="Validate with checkpoint_best.pth instead of checkpoint_final.pth.")
train_g.add_argument("-skip_val", action="store_true",
                     help="Train without the final validation pass. Note the single dash -- this "
                          "is a flag, it takes no value.")
train_g.add_argument("-val_size", type=float, default=None,
                     help="Regenerate splits_final.json with this validation fraction (0-1) "
                          "before training, instead of using nnU-Net's own 5-fold split.")
train_g.add_argument("-random_state", type=int, default=1234,
                     help="Seed for the -val_size split, so the regenerated split is "
                          "reproducible. (default: 1234)")

test_g = parser.add_argument_group("testing (--test)")
test_g.add_argument("-test_input", default="imagesTs",
                    help="Folder of test images. Resolved relative to "
                         "nnUNet_raw/<dataset_name>/ when the path does not exist as given, so "
                         "'ExtTest1_imagesTs' works. (default: imagesTs)")
test_g.add_argument("-test_input_labels", default="labelsTs",
                    help="Folder of reference labels for the evaluation, resolved the same way. "
                         "(default: labelsTs)")
test_g.add_argument("-test_output", default="",
                    help="Where predictions go. A relative name is created under "
                         "<results>/<dataset>/<tr>__<plans>__<model>/fold_<fold>/. Empty means "
                         "'imagesTs' inside that fold directory.")
test_g.add_argument("-test_summary_name", default="test_summary.json",
                    help="Metric file written next to the predictions. Give each test set its own "
                         "name so the evaluations do not overwrite each other. "
                         "(default: test_summary.json)")
test_g.add_argument("-test_checkpoint", default="checkpoint_best.pth",
                    help="Checkpoint to predict with. (default: checkpoint_best.pth)")
test_g.add_argument("--add_hd", action="store_true",
                    help="Also compute Hausdorff distances (hd95). Considerably slower than Dice "
                         "alone.")
test_g.add_argument("--overwrite_test", action="store_true",
                    help="Delete existing predictions and summary, then predict again.")
test_g.add_argument("--overwrite_test_evaluation", action="store_true",
                    help="Recompute the metrics from the predictions that are already on disk. "
                         "Use this to add --add_hd to a finished run.")
test_g.add_argument("--must_finished", action="store_true",
                    help="Require checkpoint_final.pth, i.e. refuse to evaluate a run whose "
                         "training was cut short.")
test_g.add_argument("-normalize", type=int, default=None,
                    help="Intensity-normalisation scheme passed through to code_inference.py; "
                         "leave unset to use the one recorded in the plans.")
test_g.add_argument("--multi", action="store_true",
                    help="Multi-channel/multi-input inference mode of code_inference.py.")

pre_g = parser.add_argument_group("preprocessing (--preprocess)")
pre_g.add_argument("--verify_dataset_integrity", action="store_true",
                   help="Check geometry, labels and file naming before preprocessing. Worth doing "
                        "once on a new dataset.")
pre_g.add_argument("-p_overwrite_target_spacing", default=[], type=str, nargs="+",
                   help="Force the target spacing instead of letting the planner derive it, as "
                        "three values, e.g. -p_overwrite_target_spacing 1.0 1.0 1.0.")
pre_g.add_argument("--use_nii", action="store_true",
                   help="Keep the preprocessed data as nii.gz instead of writing npy/blosc2 "
                        "arrays: much less disk, slower training.")
pre_g.add_argument("--p_use_nii", action="store_true",
                   help="Deprecated alias of --use_nii; do not pass both.")
pre_g.add_argument("--p_clean", action="store_true",
                   help="Delete the fingerprint and preprocessed folders first, forcing a clean "
                        "re-plan.")
pre_g.add_argument("--p_continue_", action="store_true",
                   help="Resume an interrupted planning step instead of restarting it.")
pre_g.add_argument("--p_continue_process", action="store_true",
                   help="Resume an interrupted preprocessing step, keeping the cases already "
                        "written.")
res_g = parser.add_argument_group("worker counts")
res_g.add_argument("-n_proc", default=12, type=int,
                   help="Worker processes for data loading and augmentation; exported as "
                        "nnUNet_def_n_proc and nnUNet_n_proc_DA. (default: 12)")
res_g.add_argument("-np", type=int, default=None,
                   help="Preprocessing workers. Unset leaves nnU-Net's own default. Each worker "
                        "holds a full volume in RAM.")
res_g.add_argument("-npfp", type=int, default=None,
                   help="Fingerprint-extraction workers. Unset leaves nnU-Net's own default.")
res_g.add_argument("-e_np", type=int, default=12,
                   help="Workers for nnUNetv2_evaluate_folder and the label-reversal step. Drop to "
                        "1 if evaluation runs out of memory. (default: 12)")

args = parser.parse_args()

os.environ['nnUNet_def_n_proc'] = str(args.n_proc)
os.environ['nnUNet_n_proc_DA'] = str(args.n_proc)


is_windows = os.name == "nt"

if args.base_dir and (args.nnunet_raw or args.nnunet_preprocessed or args.nnunet_results):
    raise ValueError("Cannot set base dir and any of the nnunet_*")

nnunet_raw = args.nnunet_raw
nnunet_results = args.nnunet_results
nnunet_preprocessed = args.nnunet_preprocessed

if args.pretrained_weights and not exists(args.pretrained_weights):
    args.pretrained_weights = join(nnunet_results, args.pretrained_weights)



os.makedirs(nnunet_results, exist_ok=True)
os.environ['nnUNet_raw'] = nnunet_raw
os.environ['nnUNet_preprocessed'] = nnunet_preprocessed
os.environ['nnUNet_results'] = nnunet_results

args.fold = str(args.fold)
device = ",".join(str(d) for d in args.device)


if args.preprocess:
    dataset_id = args.dataset_name[7:10]
    overwrite_target_spacing = f'-overwrite_target_spacing {" ".join(args.p_overwrite_target_spacing)}' if args.p_overwrite_target_spacing else ''
    command = f"{join(args.bin_path, 'nnUNetv2_plan_and_preprocess')} -d {dataset_id} -c {args.model} -pl {args.pl} {f'-np {args.np}' if args.np else ''} {f'-npfp {args.npfp}' if args.npfp else ''} {'--clean_fp --clean_pf' if args.p_clean else ''}  {'--use_nii' if args.p_use_nii or args.use_nii else ''} {'--verify_dataset_integrity' if args.verify_dataset_integrity else ''}  {'--continue_' if args.p_continue_ else ''}  {'--continue_process' if args.p_continue_process else ''}  {overwrite_target_spacing}"
    StringUtils.print(f"Doing preprocessing:\n{command}", color="green")
    os.system(command)

if args.train:
    if args.val_size is not None:
        splits_final_path = join(nnunet_preprocessed, args.dataset_name, "splits_final.json")
        if exists(splits_final_path):
            data = JsonUtils.load(splits_final_path)
            train_data, val_data = data[int(args.fold)]['train'], data[int(args.fold)]['val']
            whole_data = train_data + val_data
            train_data_, val_data_ = train_test_split(sorted(whole_data), test_size=args.val_size,
                                                      random_state=args.random_state)
            if train_data != train_data_ or val_data != val_data_:
                data[int(args.fold)] = dict(train=train_data_, val=val_data_)
                JsonUtils.dump(splits_final_path, data)
        else:
            data = [dict(), dict(), dict(), dict(), dict()]
            labels_path = join(nnunet_raw, args.dataset_name, "labelsTr")
            if exists(labels_path):
                whole_data = [item.replace(".nii.gz", "") for item in os.listdir(labels_path) if
                              item.endswith(".nii.gz")]
            else:
                whole_data = [item.replace("_0000.nii.gz", "").replace("_0001.nii.gz", "") for item in
                              os.listdir(join(nnunet_raw, args.dataset_name, "imagesTr")) if
                              item.endswith(".nii.gz")]

            train_data, val_data = train_test_split(sorted(whole_data), test_size=args.val_size, random_state=args.random_state)
            data[int(args.fold)] = dict(train=train_data, val=val_data)
            JsonUtils.dump(splits_final_path, data)

    skip_train = False
    # print(f"args.c: {args.c}")

    if args.overwrite_train and args.c:
        raise ValueError("overwrite_train and args.c cannot be true at the same time!")

    if args.c:
        dataset_res = join(nnunet_results, args.dataset_name, f"{args.tr}__{args.p}__{args.model}", f"fold_{args.fold}")
        if not exists(join(dataset_res, "checkpoint_best.pth")):
            args.c = False
            StringUtils.print(
                f"args.c was set to True while no checkpoint was available in {dataset_res}, Setting it False!")
        validation = join(dataset_res, "validation", "summary.json")
        if exists(validation):
            StringUtils.print("Validation is done skipping train")
            skip_train = True
    if not skip_train:
        if args.c and args.pretrained_weights is not None:
            print("WARNING: Since continue training is set to True then pretrain model is set to False!")
            args.pretrained_weights = ""

        if not is_windows:
            command = f"nnUNet_compile=n CUDA_VISIBLE_DEVICES={device} {join(args.bin_path, 'nnUNetv2_train')}  {args.dataset_name} {args.model} {args.fold} -device cuda -p {args.p} -tr {args.tr} {f'-batch_size {args.batch_size}' if args.batch_size else ''} {'--val' if args.val else ''} {'--skip_val' if args.skip_val else ''}  {'--val_best' if args.val_best else ''} {f'-pretrained_weights {args.pretrained_weights}' if args.pretrained_weights else ''} {'--c' if args.c else ''} -num_epochs {args.num_epochs} {'--only_encoder' if args.only_encoder else ''} -num_gpus {args.num_gpus}"
        else:
            os.environ['nnUNet_compile'] = "n"
            os.environ['CUDA_VISIBLE_DEVICES'] = f"{device}"
            command = f"nnUNetv2_train {args.dataset_name} {args.model} {args.fold} -device cuda -p {args.p} -tr {args.tr} {f'-batch_size {args.batch_size}' if args.batch_size else ''} {'--val' if args.val else ''} {'--skip_val' if args.skip_val else ''}  {'--val_best' if args.val_best else ''} {f'-pretrained_weights {args.pretrained_weights}' if args.pretrained_weights else ''} {'--c' if args.c else ''} -num_epochs {args.num_epochs} {'--only_encoder' if args.only_encoder else ''} -num_gpus {args.num_gpus}"
        os.system(command)

if args.test:
    model_path = join(nnunet_results, args.dataset_name, f"{args.tr}__{args.p}__{args.model}")
    summary_path = join(model_path, f'fold_{args.fold}', args.test_summary_name)
    if exists(args.test_input_labels):
        test_input_labels = args.test_input_labels
    else:
        test_input_labels = join(nnunet_raw, args.dataset_name, args.test_input_labels)

    if exists(args.test_input):
        test_input = args.test_input
    else:
        test_input = join(nnunet_raw, args.dataset_name, args.test_input)

    if args.test_output and exists(args.test_output):
        test_output = args.test_output
    elif args.test_output and not exists(args.test_output):
        test_output = join(model_path, f'fold_{args.fold}', args.test_output)
    else:
        test_output = join(model_path, f'fold_{args.fold}', "imagesTs")

    if args.overwrite_test:
        if exists(summary_path):
            os.remove(summary_path)
        if exists(test_output):
            shutil.rmtree(test_output)

    if not is_windows:
        command = f"nnUNet_compile=n CUDA_VISIBLE_DEVICES={device} python code_inference.py  -checkpoint {args.test_checkpoint} {f'-input {test_input}' if test_input else ''} {f'-output {test_output}' if test_output else ''} -model {args.model} -p {args.p} -device 0 -dataset_name {args.dataset_name} -tr {args.tr} -nnunet_results {nnunet_results} -nnunet_raw {nnunet_raw} -nnunet_preprocessed {nnunet_preprocessed} {f'-normalize {args.normalize}' if args.normalize else ''} {'--multi' if args.multi else ''} {'--must_finished' if args.must_finished else ''}"
    else:
        os.environ['nnUNet_compile'] = "n"
        os.environ['CUDA_VISIBLE_DEVICES'] = f"{device}"
        command = f"python code_inference.py -checkpoint {args.test_checkpoint} {f'-input {test_input}' if test_input else ''} {f'-output {test_output}' if test_output else ''} -model {args.model} -p {args.p} -device 0 -dataset_name {args.dataset_name} -tr {args.tr} -nnunet_results {nnunet_results} -nnunet_raw {nnunet_raw} -nnunet_preprocessed {nnunet_preprocessed} {f'-normalize {args.normalize}' if args.normalize else ''} {'--must_finished' if args.must_finished else ''}"

    print(f"code_inference: {command=}")
    os.system(command)

    if args.must_finished:
        final_checkpoint = join(model_path, f"fold_{args.fold}", "checkpoint_final.pth")
        if not exists(final_checkpoint):
            raise FileNotFoundError(
                f"Training is not finished, {final_checkpoint} is not available! Evaluation is aborted!")

    if exists(summary_path) and not args.overwrite_test_evaluation:
        StringUtils.print(f"{summary_path=} exists. Exit Testing ... ")
    else:
        eval_output = test_output
        dataset_json_path = join(model_path, 'dataset.json')
        reverse_class = JsonUtils.load(dataset_json_path).get('reverse_class')
        if reverse_class:
            eval_output = test_output.rstrip('/') + "_reversed"
            reverse_command = (f"nnUNet_compile=n CUDA_VISIBLE_DEVICES={device} "
                                f"{join(args.bin_path, 'nnUNetv2_reverse_normalize_folder')} {test_input}"
                                f" {test_output} {eval_output} -djfile {dataset_json_path} -np {args.e_np}")
            print(f"nnUNetv2_reverse_normalize_folder: {reverse_command=}")
            os.system(reverse_command)
        else:
            print(f"reverse_class is not set raw predictions.")

        if not is_windows:
            command = (f"nnUNet_compile=n CUDA_VISIBLE_DEVICES={device} {join(args.bin_path, 'nnUNetv2_evaluate_folder')} {test_input_labels}"
                       f" {eval_output} -djfile {join(model_path, 'dataset.json')}"
                       f" -pfile {join(model_path, 'plans.json')} -o {summary_path} {'--add_hd' if args.add_hd else ''} -np {args.e_np}")
        else:
            os.environ['nnUNet_compile'] = "n"
            os.environ['CUDA_VISIBLE_DEVICES'] = f"{device}"
            command = (f"nnUNetv2_evaluate_folder {test_input_labels}"
                       f" {eval_output} -djfile {join(model_path, 'dataset.json')}"
                       f" -pfile {join(model_path, 'plans.json')} -o {summary_path} {'--add_hd' if args.add_hd else ''} -np {args.e_np}")
        print(f"nnUNetv2_evaluate_folder: {command=}")
        os.system(command)
