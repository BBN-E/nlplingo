#!/bin/env perl

# Requirements:
#
# Run with /opt/perl-5.20.0-x86_64/bin/perl or similar
#
# git clone text-open
# cd text-open/src/java/serif ; mvn clean install
#
# git clone nlplingo

use strict;
use warnings FATAL => 'all';

# This should be used at first, due to we want to use a new enough runjobs4
use FindBin qw($Bin $Script);
use Cwd;

use File::Basename;
use File::Path;
use File::Copy;

package main;

my $textopen_root;
my $nlplingo_root;

BEGIN{
    $textopen_root = "/d4m/nlp/releases/text-open/R2020_08_20";
    # $textopen_root = "/nfs/raid88/u10/users/jcai/modern/text-open";
    $nlplingo_root = "/d4m/nlp/releases/nlplingo/R2020_08_21_1";
    # $nlplingo_root = "/home/criley/repos/nlplingo";
    # $nlplingo_root = "/nfs/raid88/u10/users/jcai/nlplingo-master/nlplingo";
    unshift(@INC, "/d4m/ears/releases/runjobs4/R2019_03_29/lib");
    unshift(@INC, "$textopen_root/src/perl/text_open/lib");
    unshift(@INC, "$nlplingo_root/perl_lib");
}

use runjobs4;
use PySerif;
use Utils;
use JSON;
use Nlplingo;

my $QUEUE_PRIO = '5'; # Default queue priority
my ($exp_root, $exp) = startjobs("queue_mem_limit" => '8G', "max_memory_over" => '0.5G', "queue_priority" => $QUEUE_PRIO);

# Parameter loading
my $params = {};
my @stages = ();
my $config_file;

if (scalar(@ARGV) < 1) {
    print "Input args that we got is EMPTY!!!!!!!!!";
    die "run.pl takes in one argument -- a config file";
}
else {
    print "Input args that we got is :";
    print join(" ", @ARGV), "\n";
    $config_file = $ARGV[0];
    $params = load_params($config_file); # $params is a hash reference
    @stages = split(/,/, get_param($params, "stages_to_run"));
    @stages = grep (s/\s*//g, @stages); # remove leading/trailing whitespaces from stage names
}

my %stages = map {$_ => 1} @stages;
my $JOB_NAME = get_param($params, "job_name");

my $LINUX_CPU_QUEUE = get_param($params, "cpu_queue", "nongale-sl6");
my $LINUX_GPU_QUEUE = get_param($params, "gpu_queue", "allGPUs-sl610-non-k10s");

# Location of all the output of this sequence
(my $processing_dir, undef) = Utils::make_output_dir("$exp_root/expts/$JOB_NAME", "$JOB_NAME/mkdir_job_directory", []);

# Make copy of config file for debugging purposes
copy($config_file, $processing_dir . "/" . get_current_time() . "-" . basename($config_file));

# Python commands
my $PYTHON3 = "/opt/Python-3.5.2-x86_64/bin/python3.5 -u";
my $ANACONDA_ROOT = "";
if (get_param($params, "ANACONDA_ROOT", "None") eq "None") {
    $ANACONDA_ROOT = "/nfs/raid87/u11/users/hqiu/miniconda_prod";
}
else {
    $ANACONDA_ROOT = get_param($params, "ANACONDA_ROOT");
}

my $CREATE_FILELIST_PY_PATH = "$textopen_root/src/python/util/common/create_filelist_with_batch_size.py";
my $CREATE_FILE_LIST_SCRIPT = "$PYTHON3 $textopen_root/src/python/util/common/create_filelist.py";

my $CONDA_ENV_NAME_FOR_NLPLINGO_CPU = "py3-jni";
my $CONDA_ENV_NAME_FOR_NLPLINGO_GPU = "nlplingo-gpu";

# Exes
my $BASH = "/bin/bash";

my $only_cpu_available = (get_param($params, "only_cpu_available", "false") eq "true");
if ($only_cpu_available) {
    $CONDA_ENV_NAME_FOR_NLPLINGO_GPU = $CONDA_ENV_NAME_FOR_NLPLINGO_CPU
}

my $NUM_OF_BATCHES_GLOBAL = get_param($params, "num_of_batches_global", 1);
my $IN_MEMORY = get_param($params, "in_memory", 0);
my $TRAIN_CANONICAL_TEMPLATE_DIR = get_param($params, "train_canonical_template_dir", "");
if ($TRAIN_CANONICAL_TEMPLATE_DIR eq "") {
    $TRAIN_CANONICAL_TEMPLATE_DIR = "$nlplingo_root/sequences/templates/canonical/train";
}
my $TEST_CANONICAL_TEMPLATE_DIR = get_param($params, "test_canonical_template_dir", "");
if ($TEST_CANONICAL_TEMPLATE_DIR eq "") {
    $TEST_CANONICAL_TEMPLATE_DIR = "$nlplingo_root/sequences/templates/canonical/test";
}
my $DECODE_CANONICAL_TEMPLATE_DIR = get_param($params, "decode_canonical_template_dir", "");
if ($DECODE_CANONICAL_TEMPLATE_DIR eq "") {
    $DECODE_CANONICAL_TEMPLATE_DIR = "$nlplingo_root/sequences/templates/canonical/decode";
}
# Generally should be identical to train, but with save_model = false.
# TODO just add +save_model+ expandable parameter to training template and modify all usages
my $TUNE_CANONICAL_TEMPLATE_DIR = get_param($params, "tune_canonical_template_dir", "");
if ($TUNE_CANONICAL_TEMPLATE_DIR eq "") {
    $TUNE_CANONICAL_TEMPLATE_DIR = "$nlplingo_root/sequences/templates/canonical/tune";
}

my $WITH_TUNING = exists $stages{"tune"};
my $number_of_tuning_folds = get_param($params, "k_folds", "");  # default: no cross-validation

check_requirements();

# Parse comma-separated list of tasks
my @tasks = ();
@tasks = split(/,/, get_param($params, "tasks"));
@tasks = grep (s/\s*//g, @tasks); # remove leading/trailing whitespaces from stage names

my $nlplingo_obj = Nlplingo->new(
    PYTHON_CPU        => "$ANACONDA_ROOT/envs/$CONDA_ENV_NAME_FOR_NLPLINGO_CPU/bin/python",
    PYTHON_GPU        => "$ANACONDA_ROOT/envs/$CONDA_ENV_NAME_FOR_NLPLINGO_GPU/bin/python",
    SERIF_PYTHON_PATH => "$textopen_root/src/python",
    JOB_NAME          => "$JOB_NAME/nlplingo_object",
    LINUX_CPU_QUEUE   => "$LINUX_CPU_QUEUE",
    LINUX_GPU_QUEUE   => "$LINUX_GPU_QUEUE",
    USE_GPU           => !$only_cpu_available,
    SGE_VIRTUAL_FREE  => ['32G','64G','192G']
);

########
# Run preprocessing (serialization of nlplingo Datapoints to disk) for multiple tasks
########
my $serialized_file_parent_dir; # used for reconstruction later
my %serialize_job_ids;
my %tuned_params;
if (exists $stages{"preprocess"} and not($IN_MEMORY)) {

    my $preprocess_template = {
        BATCH_QUEUE                        => $LINUX_CPU_QUEUE,
        SGE_VIRTUAL_FREE                   => "10G",
        max_memory_over                    => "50G"
    };

    my $stage_name = "preprocess";

    my @dep_aggregate_list_ids = ();

    # Concatenate the SerifXML train/dev/test lists
    my $aggregate_serif_output_dir;
    {
        my $mini_stage_name = "aggregate_lists";
        (my $ministage_processing_dir, undef) = Utils::make_output_dir("$processing_dir/$stage_name/$mini_stage_name/", "$JOB_NAME/$stage_name/$mini_stage_name/mkdir_stage_processing/", []);
        foreach my $task (@tasks) {
            my $train_json = $TRAIN_CANONICAL_TEMPLATE_DIR . "/$task" . ".json";
            # my $train_json = get_param($params, $task);

            my $list_create_dep_jobid = runjobs4::runjobs(
            [], "$JOB_NAME/$stage_name/$mini_stage_name/$task/aggregate_lists",
            $preprocess_template,
            [ "env PYTHONPATH=$nlplingo_root:$textopen_root/src/python " .
                "KERAS_BACKEND=tensorflow " .
                "$ANACONDA_ROOT/envs/$CONDA_ENV_NAME_FOR_NLPLINGO_CPU/bin/python $nlplingo_root/nlplingo/modes/preprocess/generate_lists_and_mode_maps.py $train_json $task $ministage_processing_dir"]
            );
            push(@dep_aggregate_list_ids, $list_create_dep_jobid);
        }
        $aggregate_serif_output_dir = $ministage_processing_dir;
    }

    # Split the concatenated SerifXML list into multiple batches (for parallelization)
    # And write the nlplingo Datapoints (with populated features) onto disk
    {
        my $mini_stage_name = "serialize";
        foreach my $task (@tasks) {
            $serialize_job_ids{$task} = ();
            my $train_json = $TRAIN_CANONICAL_TEMPLATE_DIR . "/$task" . ".json";
            (my $ministage_processing_dir, undef) = Utils::make_output_dir("$processing_dir/$stage_name/$mini_stage_name/$task", "$JOB_NAME/$stage_name/$mini_stage_name/mkdir_stage_processing/$task", []);
            (my $ministage_batch_dir, my $mkdir_batch_jobs) = Utils::make_output_dir("$processing_dir/$stage_name/$mini_stage_name/$task/batch_file", "$JOB_NAME/$stage_name/$mini_stage_name/$task/mkdir_stage_batch_processing", \@dep_aggregate_list_ids);


            (my $create_filelist_jobid, undef
            ) = Utils::split_file_list_with_num_of_batches(
                PYTHON                  => $PYTHON3,
                CREATE_FILELIST_PY_PATH => $CREATE_FILELIST_PY_PATH,
                dependant_job_ids       => $mkdir_batch_jobs,
                job_prefix              => "$JOB_NAME/$stage_name/$mini_stage_name/$task",
                num_of_batches          => $NUM_OF_BATCHES_GLOBAL,
                list_file_path          => "$aggregate_serif_output_dir/$task" . ".list",
                output_file_prefix      => $ministage_batch_dir . "/batch_" . $task. "_",
                suffix                  => ".list",
            );


            # Serialize the nlplingo Datapoints batch by batch
            for (my $batch = 0; $batch < $NUM_OF_BATCHES_GLOBAL; $batch++) {
                my $batch_file = "$ministage_batch_dir/batch_" . $task . "_" . $batch . ".list";
                my $batch_output_folder = "$ministage_processing_dir/$batch/output";
                my $serialize_dep_jobid = runjobs4::runjobs(
                $mkdir_batch_jobs, "$JOB_NAME/$stage_name/$mini_stage_name/" . $task . "/serialize_$batch",
                $preprocess_template,
                [ "mkdir -p $batch_output_folder" ],
                [ "env PYTHONPATH=$nlplingo_root:$textopen_root/src/python " .
                    "KERAS_BACKEND=tensorflow " .
                    "$ANACONDA_ROOT/envs/$CONDA_ENV_NAME_FOR_NLPLINGO_CPU/bin/python  $nlplingo_root/nlplingo/modes/preprocess/serialize.py $train_json $task $batch_file $batch_output_folder $aggregate_serif_output_dir $NUM_OF_BATCHES_GLOBAL"]
                );
                push(@{$serialize_job_ids{$task} }, $serialize_dep_jobid);
            }
        }
        $serialized_file_parent_dir = "$processing_dir/$stage_name/$mini_stage_name";
    }
}

my %train_modes =  (  sequence => "train_ner",
                event_argument => "train_argument",
                entity_coref => "train_entitycoref",
                event_coref => "train_eventcoref",
                entity_relation => "train_entity_relation_from_file",
                event_relation => "train_evevr_from_file",
                event_trigger => "train_trigger_from_file",
                event_relation_pytorch => "train_eer_pytorch",
                entity_relation_bert_mention => "train_entity_bert_mention",
                event_argument_bert_mention => "train_argument_bert_mention",
                sequence => "train_ner"
             );

my %test_modes =  (  sequence => "test_ner",
);

my %decode_modes =  (  sequence => "decode_ner",
);

my %job_core_array =  (  sequence => 1,
                event_argument => 1,
                entity_coref => 1,
                event_coref => 1,
                entity_relation => 1,
                event_relation => 1,
                event_trigger => 1,
                event_relation_pytorch => 2,
                entity_relation_bert_mention => 2,
                event_argument_bert_mention => 2,
                sequence => 2
             );


if ($WITH_TUNING) {

    my $stage_name = "tune";
    if ($IN_MEMORY) {
        die "Cannot run '$stage_name' stage in memory!";
    }

    my %task_to_tune_method = (
        "sequence"       => "parameter_search_sequence",
        "event_trigger"  => "parameter_search_trigger",
        "event_argument" => "parameter_search_argument",
    );

    my %task_to_search_jobs = ();
    my $parameter_search_ministage_dir;
    {
        my $mini_stage_name = "parameter_search";
        unshift(@INC, get_param($params, "tuning_params_dir"));
        foreach my $task (@tasks) {

            my $tune_method;
            if (!exists $task_to_tune_method{$task}) {
                next;
            }
            else {
                $tune_method = $task_to_tune_method{$task};
            }

            (my $ministage_processing_dir, undef) = Utils::make_output_dir(
                "$processing_dir/$stage_name/$mini_stage_name/$task",
                "$JOB_NAME/$stage_name/$mini_stage_name/mkdir_stage_processing/$task",
                []
            );

            # Tuning a model depends on previous serialize jobs.
            # Collect the serialized files into a list.
            my @serialize_jobs = $serialize_job_ids{$task};
            my $serialize_npz_list = "$serialized_file_parent_dir/$task/" . "npzs_for_tune.list";
            my $list_npz_jobid = generate_file_list(
                @serialize_jobs,
                "$JOB_NAME/$stage_name/$mini_stage_name/list-npz-$task",
                "$serialized_file_parent_dir/$task/*/output/*.npz",
                $serialize_npz_list);

            # Prepare configurations for tuning.
            my $tuning_params = "$task" . "_tune";
            require "$tuning_params.pm";
            # my $tuning_params_obj = $tuning_params->new();
            my %params_to_search = $tuning_params->params_to_search();
            my %param_to_abbrev = $tuning_params->param_to_abbrev();
            my %params_override = ();

            my $parameter_search_jobs = $nlplingo_obj->$tune_method(
                dependant_job_ids  => [ $list_npz_jobid ],
                param_file         => "$TUNE_CANONICAL_TEMPLATE_DIR/$task.json",
                params_to_search   => \%params_to_search,
                param_to_abbrev    => \%param_to_abbrev,
                param_override     => \%params_override,
                output_dir         => $ministage_processing_dir,
                serialized_list    => $serialize_npz_list,
                serialized_k_folds => $number_of_tuning_folds
            );
            $task_to_search_jobs{$task} = $parameter_search_jobs;
        }
        $parameter_search_ministage_dir = "$processing_dir/$stage_name/$mini_stage_name";
    }

    {
        my $mini_stage_name = "select";
        foreach my $task (@tasks) {

            if (!exists $task_to_search_jobs{$task}) {
                next;
            }
            my $scores_have_been_aggregated = "$number_of_tuning_folds" ne "";

            (my $ministage_processing_dir, undef) = Utils::make_output_dir(
                "$processing_dir/$stage_name/$mini_stage_name/$task",
                "$JOB_NAME/$stage_name/$mini_stage_name/mkdir_stage_processing/$task",
                []
            );

            my %base_params_override = (
                # none of these parameters are used when running from serialized
                # but they exist in templates/base_tune_params.json and need to
                # be defined in the override hash
                train_file_list           => "dummy",
                test_file_list            => "dummy",
                dev_file_list             => "dummy",
                negative_trigger_words    => "/nfs/raid88/u10/users/ychan/ace_experiments/resources/negative_trigger_words",
                add_serif_event_mentions  => "false",
                add_serif_entity_mentions => "false",
            );

            my $train_json = $TRAIN_CANONICAL_TEMPLATE_DIR . "/$task" . ".json";
            my ($select_model_jobs, $best_params_json) = $nlplingo_obj->find_best_params_file(
                aggregated => $scores_have_been_aggregated,
                dependant_job_ids => $task_to_search_jobs{$task},
                base_params_file => "$train_json",
                base_param_override => \%base_params_override,
                instance_name => $task,
                input_dir => "$parameter_search_ministage_dir/$task",
                output_dir => $ministage_processing_dir
            );

            # Uncomment to force training stage to wait for tuning stage.
            push(@{$serialize_job_ids{$task} }, @$select_model_jobs);
            
            $tuned_params{$task} = $best_params_json;
        }
    }
}


# Train a model
# Most models use a dev set and emit dev/test scores
my %model_file_ids;
my %model_file_paths;
if (exists $stages{"train"}) {
    my %train_job_ids;
    if ($IN_MEMORY) {
        my $stage_name = "in_memory_end_to_end_run";
        # Train/test, doing everything in memory (including featurization of nlplingo Datapoints)
        {
            my $mini_stage_name = "train_test";

            foreach my $task (@tasks) {
                # 1. Train the model
                my $train_json = $TRAIN_CANONICAL_TEMPLATE_DIR . "/$task" . ".json";
                (my $ministage_processing_dir, my $mkdir_job_id) = Utils::make_output_dir("$processing_dir/$stage_name/$mini_stage_name/$task", "$JOB_NAME/$stage_name/$mini_stage_name/mkdir_stage_processing/$task", []);

                my $train_template = {
                    BATCH_QUEUE                        => $LINUX_GPU_QUEUE,
                    SGE_VIRTUAL_FREE                   => "80G",
                    output_dir                         => $ministage_processing_dir,
                    job_cores                          => $job_core_array{$task} 
                };

                my $nlplingo_cmd = $nlplingo_obj->train_command(
                    params_template => $train_json,
                    mode            => $train_modes{$task});
                my $train_model_id = runjobs4::runjobs(
                    $mkdir_job_id,
                    "$JOB_NAME/$stage_name/$mini_stage_name/$task",
                    $train_template,
                    $nlplingo_cmd
                );
                # $train_job_ids{$task} = ();
                # push(@{$train_job_ids{$task} }, $train_model_id);

                # 2. Write the trained model path to a file, which can be used later at test/decode stages. 
                my $model_path_template = {
                    BATCH_QUEUE                        => $LINUX_CPU_QUEUE,
                    SGE_VIRTUAL_FREE                   => "1G",
                    output_dir                         => $ministage_processing_dir
                };
                my $model_file_id = runjobs4::runjobs(
                $train_model_id, "$JOB_NAME/$stage_name/$mini_stage_name/model_file_write_$task",
                $model_path_template,
                [ "env PYTHONPATH=$nlplingo_root:$textopen_root/src/python " .
                    "KERAS_BACKEND=tensorflow " .
                    "$ANACONDA_ROOT/envs/$CONDA_ENV_NAME_FOR_NLPLINGO_CPU/bin/python  $nlplingo_root/nlplingo/tasks/common/util/write_model_file.py --params ", $train_json, "--task $task --output_directory $ministage_processing_dir"]);
                $model_file_ids{$task} = $model_file_id;
                $model_file_paths{$task} = "$ministage_processing_dir/$task" . ".model_file";
            }
        }
    } else {
        my $stage_name = "from_serialized";
        # Train/test from the serialized nlplingo Datapoints serialized earlier
        {
            my $mini_stage_name = "train_test";

            foreach my $task (@tasks) {
                # 1. Train the model
                
                # Training a model in this mode depends on previous serialize jobs.
                my @serialize_jobs = $serialize_job_ids{$task};

                # Collect the serialized files into a list.
                my $serialize_npz_list = "$serialized_file_parent_dir/$task/" . "npzs.list";
                my $list_npz_jobid = generate_file_list(@serialize_jobs, "$JOB_NAME/$stage_name/$mini_stage_name/list-npz-$task", "$serialized_file_parent_dir/$task/*/output/*.npz", $serialize_npz_list);

                # Train the model; note the --serialize_list argument, which differs from before.
                (my $ministage_processing_dir, my $mkdir_job_id) = Utils::make_output_dir("$processing_dir/$stage_name/$mini_stage_name/$task", "$JOB_NAME/$stage_name/$mini_stage_name/mkdir_stage_processing/$task", $list_npz_jobid);
                my $train_template = {
                    BATCH_QUEUE                        => $LINUX_GPU_QUEUE,
                    SGE_VIRTUAL_FREE                   => "80G",
                    output_dir                         => $ministage_processing_dir,
                    job_cores                          => $job_core_array{$task} 
                };

                my $train_json = $TRAIN_CANONICAL_TEMPLATE_DIR . "/$task" . ".json";

                # # this block works, but only if you run the tuning stage and
                # # training stage in separate (sequential) processes with the
                # # same job name.  It's more confusing than it's worth.
                # my $train_json;
                # if (exists $tuned_params{$task}){
                #     $train_json = $tuned_params{$task};
                # } else {
                #      $train_json = $TRAIN_CANONICAL_TEMPLATE_DIR . "/$task" . ".json";
                # }

                my @training_prereqs = ();
                push @training_prereqs, @$mkdir_job_id;
                my $train_command_arrayref = $nlplingo_obj->train_command(
                    params_template => $train_json,
                    mode            => $train_modes{$task},
                    serialized      => $serialize_npz_list
                );
                if ($WITH_TUNING) {

                    # Can't use a template for the params, because templates are
                    # expanded when the sequence starts.
                    my $modify_best_params_job = runjobs4::runjobs(
                        @training_prereqs,
                        "$JOB_NAME/$stage_name/modify_best_params/$mini_stage_name/$task",
                        {
                            BATCH_QUEUE      => $LINUX_CPU_QUEUE,
                            SGE_VIRTUAL_FREE => "1G",
                            output_dir       => $ministage_processing_dir,
                            search_str       => "$JOB_NAME/tune/.\\+/$task",
                            params           => "$processing_dir/tune/select/$task/best.params.json"
                        },
                        [ "$BASH", "switch_output_dirs.sh" ]
                    );

                    push @training_prereqs, $modify_best_params_job;

                    $train_command_arrayref = $nlplingo_obj->train_command(
                        params_json     => "$ministage_processing_dir/train.json",
                        mode            => $train_modes{$task},
                        serialized      => $serialize_npz_list
                    );
                }

                my $train_model_id = runjobs4::runjobs(
                    \@training_prereqs,
                    "$JOB_NAME/$stage_name/$mini_stage_name/$task",
                    $train_template,
                    $train_command_arrayref
                );
                # $train_job_ids{$task} = ();
                # push(@{$train_job_ids{$task} }, $train_model_id);

                # 2. Write the trained model path to a file, which can be used later at test/decode stages.
                my $model_path_template = {
                    BATCH_QUEUE                        => $LINUX_CPU_QUEUE,
                    SGE_VIRTUAL_FREE                   => "1G",
                    output_dir                         => $ministage_processing_dir
                };
                my $model_file_id = runjobs4::runjobs(
                $train_model_id, "$JOB_NAME/$stage_name/$mini_stage_name/model_file_write_$task",
                $model_path_template,
                [ "env PYTHONPATH=$nlplingo_root:$textopen_root/src/python " .
                    "KERAS_BACKEND=tensorflow " .
                    "$ANACONDA_ROOT/envs/$CONDA_ENV_NAME_FOR_NLPLINGO_CPU/bin/python  $nlplingo_root/nlplingo/tasks/common/util/write_model_file.py --params ", $train_json, "--task $task --output_directory $ministage_processing_dir"]);
                $model_file_ids{$task} = $model_file_id;
                $model_file_paths{$task} = "$ministage_processing_dir/$task" . ".model_file";
            }
        }
    }
}

dojobs();

# Emit a score on a test set for a trained model
if (exists $stages{"test"}) {
    my $stage_name = "test";
    {
        my $mini_stage_name = "score";
        foreach my $task (@tasks) {
            if (exists($test_modes{$task})) {
                # Read the model path in
                my $model_file_path = $model_file_paths{$task};
                open my $file, '<',  $model_file_path; 
                my $actual_model = <$file>; 
                close $file;

                my $test_json = $TEST_CANONICAL_TEMPLATE_DIR .  "/$task" . ".json";
                (my $ministage_processing_dir, my $mkdir_job_id) = Utils::make_output_dir("$processing_dir/$stage_name/$mini_stage_name/$task", "$JOB_NAME/$stage_name/$mini_stage_name/mkdir_stage_processing/$task", []);
                my $model_file_job = $model_file_ids{$task};

                my $test_template;
                if ($task eq "sequence") {
                    # Sequence task does not finish in reasonable amount of time without GPU
                    # So use GPU
                    $test_template = {
                        BATCH_QUEUE                        => $LINUX_GPU_QUEUE,
                        SGE_VIRTUAL_FREE                   => "20G",
                        output_dir                         => $ministage_processing_dir,
                        model_file                         => $actual_model,
                        job_cores                          => 1,
                    };
                } else {
                    # CPU
                    $test_template = {
                        BATCH_QUEUE                        => $LINUX_CPU_QUEUE,
                        SGE_VIRTUAL_FREE                   => "20G",
                        output_dir                         => $ministage_processing_dir,
                        model_file                         => $actual_model
                    };
                }

                my $nlplingo_cmd = $nlplingo_obj->train_command(
                    params_template => $test_json,
                    mode            => $test_modes{$task});
                my $test_id = runjobs4::runjobs(
                    $model_file_job,
                    "$JOB_NAME/$stage_name/$mini_stage_name/test_$task",
                    $test_template,
                    ["echo $test_json"],  # where is this template from?
                    $nlplingo_cmd
                );
            }
        }
    }
}

# Decode over SerifXML for a trained model
if (exists $stages{"decode"}) {
    my $stage_name = "decode";
    {
        my $mini_stage_name = "score";
        foreach my $task (@tasks) {
            if (exists($decode_modes{$task})) {
                # Read the model path in
                my $model_file_path = $model_file_paths{$task};
                open my $file, '<',  $model_file_path; 
                my $actual_model = <$file>; 
                close $file;

                my $decode_json = $DECODE_CANONICAL_TEMPLATE_DIR .  "/$task" . ".json";
                (my $ministage_processing_dir, my $mkdir_job_id) = Utils::make_output_dir("$processing_dir/$stage_name/$mini_stage_name/$task", "$JOB_NAME/$stage_name/$mini_stage_name/mkdir_stage_processing/$task", []);
                my $model_file_job = $model_file_ids{$task};

                my $test_template;
                if ($task eq "sequence") {
                    # Sequence task does not finish in reasonable amount of time without GPU
                    # So use GPU
                    $test_template = {
                        BATCH_QUEUE                        => $LINUX_GPU_QUEUE,
                        SGE_VIRTUAL_FREE                   => "20G",
                        output_dir                         => $ministage_processing_dir,
                        model_file                         => $actual_model,
                        job_cores                          => 1,
                    };

                } else {
                    # CPU
                    $test_template = {
                        BATCH_QUEUE                        => $LINUX_CPU_QUEUE,
                        SGE_VIRTUAL_FREE                   => "20G",
                        output_dir                         => $ministage_processing_dir,
                        model_file                         => $actual_model
                    };

                }

                my $nlplingo_cmd = $nlplingo_obj->train_command(
                    params_template => $decode_json,
                    mode            => $decode_modes{$task});
                my $test_id = runjobs4::runjobs(
                    $model_file_job,
                    "$JOB_NAME/$stage_name/$mini_stage_name/decode_$task",
                    $test_template,
                    $nlplingo_cmd
                );
            }
        }
    }
}

endjobs();

sub load_params {
    my %params = ();
    my $config_file = $_[0];

    open(my $fh, '<', $config_file) or die "Could not open config file: $config_file";
    while (my $line = <$fh>) {
        $line =~ s/^\s+|\s+$//g;
        next if length($line) == 0;
        next if substr($line, 0, 1) eq '#';
        my @pieces = split(/:/, $line, 2);
        if (scalar(@pieces) != 2) {
            die "Could not find key: value pair in config file line: $line\n";
        }
        my $param_name = $pieces[0];
        my $param_value = $pieces[1];
        $param_name =~ s/^\s+|\s+$//g;
        $param_value =~ s/^\s+|\s+$//g;
        $params{$param_name} = $param_value;
    }

    close($fh);

    return \%params;
}

sub get_param {
    my $params_ref = $_[0];
    my $param_key = $_[1];
    my $default_value;

    if (scalar(@_) > 2) {
        $default_value = $_[2];
    }

    if (!defined($params_ref->{$param_key})) {
        if (defined($default_value)) {
            return $default_value;
        }
        else {
            die "Required parameter: $param_key not set";
        }
    }

    return $params_ref->{$param_key};
}

sub check_requirements {

    my @required_git_repos = ();

    for my $git_repo (@required_git_repos) {
        if (!-e $git_repo) {
            die "You must have the git repo: " . $git_repo . " cloned\n";

        }

    }
}

sub get_current_time {
    my ($sec, $min, $hour, $mday, $mon, $year, $wday, $yday, $isdst) = localtime(time);
    my $nice_timestamp = sprintf("%04d%02d%02d-%02d%02d%02d",
        $year + 1900, $mon + 1, $mday, $hour, $min, $sec);
    return $nice_timestamp;
}

sub generate_file_list {
    my @job_dependencies = @{$_[0]};
    my $create_list_job_name = $_[1];
    my $unix_path_str = $_[2];
    my $output_file_path = $_[3];
    return runjobs(
        \@job_dependencies, $create_list_job_name,
        {
            SCRIPT => 1
        },
        [ "$CREATE_FILE_LIST_SCRIPT --unix_style_pathname \"$unix_path_str\" --output_list_path $output_file_path" ]
    );
}
