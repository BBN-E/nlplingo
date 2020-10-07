#!/bin/env perl

# Requirements:
#
# Run with /opt/perl-5.20.0-x86_64/bin/perl or similar
# Use Java 8 -- export JAVA_HOME="/opt/jdk1.8.0_20-x86_64"
# environment variable SVN_PROJECT_ROOT (in .bashrc) should point to Active/Projects where SERIF/python, SERIF/par, W-ICEWS/lib are checked out
# If you have a "# Stop here for an non-interactive shell." section in your .bashrc file, make sure the relevant environment variables (above) are above that section to make sure runjobs can see them
#
# git clone text-open
# cd text-open/src/java/serif ; mvn clean install
#
# git clone jserif
# cd jserif ; mvn clean install
#
# cd Hume/src/java/serif-util ; mvn clean install -DskipTests
#
# git clone learnit
# cd learnit ; mvn clean install
#
# git clone kbp
# git clone deepinsight
# git clone nlplingo
#

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

my $jserif_event_jar;
BEGIN{
    $textopen_root = "/d4m/nlp/releases/text-open/R2020_08_20";
    # $textopen_root = "/nfs/raid88/u10/users/jcai/modern/text-open";
    # $nlplingo_root = "/d4m/nlp/releases/nlplingo/R2020_06_01";
    # $nlplingo_root = "/d4m/nlp/releases/nlplingo/R2020_08_20_1"; 
    $nlplingo_root = "/nfs/raid88/u10/users/jcai/nlplingo-master/nlplingo";
    $jserif_event_jar = "/d4m/nlp/releases/jserif/serif-event/serif-events-8.10.3-SNAPSHOT-pg.jar"; # For kbp
    unshift(@INC, "/d4m/ears/releases/runjobs4/R2019_03_29/lib");
    unshift(@INC, "$textopen_root/src/perl/text_open/lib");
}

use runjobs4;
use PySerif;
use Utils;

my $QUEUE_PRIO = '5'; # Default queue priority
my ($exp_root, $exp) = startjobs("queue_mem_limit" => '8G', "max_memory_over" => '0.5G', "queue_priority" => $QUEUE_PRIO);

# Parameter loading
my $params = {};
my @stages = ();
if (scalar(@ARGV) < 1) {
    print "Input args that we got is EMPTY!!!!!!!!!";
    die "run.pl takes in one argument -- a config file";
}
else {
    print "Input args that we got is :";
    print join(" ", @ARGV), "\n";
    my $config_file = $ARGV[0];
    $params = load_params($config_file); # $params is a hash reference
    @stages = split(/,/, get_param($params, "stages_to_run"));
    @stages = grep (s/\s*//g, @stages); # remove leading/trailing whitespaces from stage names
}

my %stages = map {$_ => 1} @stages;
my $JOB_NAME = get_param($params, "job_name");

my $TRAIN_FILE_BERT = get_param($params, "input_train_file_bert", "");
my $VAL_FILE_BERT = get_param($params, "input_val_file_bert", "");
my $NUM_EPOCH_BERT = get_param($params, "num_epoch_bert", 1);
my $ONTOLOGY_BERT = get_param($params, "ontology_bert", "");

my $TRAIN_FILE_CNN = get_param($params, "input_train_file_cnn", "");
my $VAL_FILE_CNN = get_param($params, "input_val_file_cnn", "");
my $NUM_EPOCH_CNN = get_param($params, "num_epoch_cnn", 1);
my $ONTOLOGY_CNN = get_param($params, "ontology_cnn", "");


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

my $CONDA_ENV_NAME_FOR_DOC_RESOLVER = "py3-jni";
my $CONDA_ENV_NAME_FOR_BERT_CPU = "py3-jni";
my $CONDA_ENV_NAME_FOR_BERT_GPU = "p3-bert-gpu";
my $CONDA_ENV_NAME_FOR_NLPLINGO_GPU = "nlplingo-gpu";
my $CONDA_ENV_NAME_FOR_NN_EVENT_TYPING = "python-tf0.11-cpu";

# Location of all the output of this sequence
(my $processing_dir, undef) = Utils::make_output_dir("$exp_root/expts/$JOB_NAME", "$JOB_NAME/mkdir_job_directory", []);

my $LINUX_QUEUE = get_param($params, "cpu_queue", "nongale-sl6");
my $LINUX_GPU_QUEUE = get_param($params, "gpu_queue", "allGPUs-p100");

################
# Train nlplingo model
################
if (exists $stages{"train_ldc_bert"}) {
    my $stage_name = "train_ldc_bert";

    {
        my $mini_stage_name = "train";

        my $train_id;
        (my $ministage_processing_dir, my $mkdir_job_id) = Utils::make_output_dir("$processing_dir/$stage_name/$mini_stage_name/", "$JOB_NAME/$stage_name/$mini_stage_name/mkdir_stage_processing", []);

        my $pyserif_template = {
            BATCH_QUEUE                        => $LINUX_GPU_QUEUE,
            SGE_VIRTUAL_FREE                   => "80G",
            input_train_file                   => $TRAIN_FILE_BERT,
            input_val_file                     => $VAL_FILE_BERT,
            ontology_file                      => $ONTOLOGY_BERT,
            num_epoch                          => $NUM_EPOCH_BERT,
            save_model_path                    => $ministage_processing_dir,
            job_cores                          => 2
        };

        my $eer_train_template = "train_bert_mentionpool_prod.json";

        my $train_model_id = runjobs4::runjobs(
        $mkdir_job_id, "$JOB_NAME/$stage_name/$mini_stage_name/",
        $pyserif_template,
        [ "env PYTHONPATH=$nlplingo_root:$textopen_root/src/python " .
            "KERAS_BACKEND=tensorflow " .
            "$ANACONDA_ROOT/envs/$CONDA_ENV_NAME_FOR_NLPLINGO_GPU/bin/python $nlplingo_root/nlplingo/tasks/train_test.py --params", $eer_train_template, "--mode train_eer_from_txt"]
        );
    }

    my $create_decoding_id;
    {
        # create decoding config file
        my $mini_stage_name = "create_decoding";
        (my $ministage_processing_dir, my $mkdir_job_id) = Utils::make_output_dir("$processing_dir/$stage_name/$mini_stage_name", "$JOB_NAME/$stage_name/$mini_stage_name/mkdir_stage_processing", []);

        #my $pyserif_template = {
        #    BATCH_QUEUE                        => $LINUX_QUEUE,
        #    SGE_VIRTUAL_FREE                   => "8G",
        #    ontology_file                      => $ontology_path,
        #    decoding_threshold                 => $DECODING_THRESHOLD
        #};

        #my $eer_test_template = "test_bert_mentionpool_prod.json";

        #$create_decoding_id = runjobs4::runjobs(
        #$mkdir_job_id, "$JOB_NAME/$stage_name/$mini_stage_name/",
        #$pyserif_template,
        #[ "env PYTHONPATH=$nlplingo_root:$textopen_root/src/python " .
        #    "KERAS_BACKEND=tensorflow " .
        #    "$ANACONDA_ROOT/envs/$CONDA_ENV_NAME_FOR_NLPLINGO_GPU/bin/python $nlplingo_root/nlplingo/tasks/eventrelation/create_decoding_config.py", $eer_test_template, "$fuse_test_score_list $ministage_processing_dir"]
        #);
    }
}

if (exists $stages{"train_giga_cnn"}) {
    my $stage_name = "train_giga_cnn";

    {
        my $mini_stage_name = "train";

        my $train_id;
        (my $ministage_processing_dir, my $mkdir_job_id) = Utils::make_output_dir("$processing_dir/$stage_name/$mini_stage_name/", "$JOB_NAME/$stage_name/$mini_stage_name/mkdir_stage_processing", []);

        my $pyserif_template = {
            BATCH_QUEUE                        => $LINUX_GPU_QUEUE,
            SGE_VIRTUAL_FREE                   => "160G",
            input_train_file                   => $TRAIN_FILE_CNN,
            input_val_file                     => $VAL_FILE_CNN,
            ontology_file                      => $ONTOLOGY_CNN,
            num_epoch                          => $NUM_EPOCH_CNN,
            save_model_path                    => $ministage_processing_dir,
            job_cores                          => 2
        };

        my $eer_train_template = "train_cnn_prod.json";

        my $train_model_id = runjobs4::runjobs(
        $mkdir_job_id, "$JOB_NAME/$stage_name/$mini_stage_name/",
        $pyserif_template,
        [ "env PYTHONPATH=$nlplingo_root:$textopen_root/src/python " .
            "KERAS_BACKEND=tensorflow " .
            "$ANACONDA_ROOT/envs/$CONDA_ENV_NAME_FOR_NLPLINGO_GPU/bin/python $nlplingo_root/nlplingo/tasks/train_test.py --params", $eer_train_template, "--mode train_eer_from_txt"]
        );
    }

    my $create_decoding_id;
    {
        # create decoding config file
        my $mini_stage_name = "create_decoding";
        (my $ministage_processing_dir, my $mkdir_job_id) = Utils::make_output_dir("$processing_dir/$stage_name/$mini_stage_name", "$JOB_NAME/$stage_name/$mini_stage_name/mkdir_stage_processing", []);

        #my $pyserif_template = {
        #    BATCH_QUEUE                        => $LINUX_QUEUE,
        #    SGE_VIRTUAL_FREE                   => "8G",
        #    ontology_file                      => $ontology_path,
        #    decoding_threshold                 => $DECODING_THRESHOLD
        #};

        #my $eer_test_template = "test_bert_mentionpool_prod.json";

        #$create_decoding_id = runjobs4::runjobs(
        #$mkdir_job_id, "$JOB_NAME/$stage_name/$mini_stage_name/",
        #$pyserif_template,
        #[ "env PYTHONPATH=$nlplingo_root:$textopen_root/src/python " .
        #    "KERAS_BACKEND=tensorflow " .
        #    "$ANACONDA_ROOT/envs/$CONDA_ENV_NAME_FOR_NLPLINGO_GPU/bin/python $nlplingo_root/nlplingo/tasks/eventrelation/create_decoding_config.py", $eer_test_template, "$fuse_test_score_list $ministage_processing_dir"]
        #);
    }
}



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

sub get_current_time {
    my ($sec, $min, $hour, $mday, $mon, $year, $wday, $yday, $isdst) = localtime(time);
    my $nice_timestamp = sprintf("%04d%02d%02d-%02d%02d%02d",
        $year + 1900, $mon + 1, $mday, $hour, $min, $sec);
    return $nice_timestamp;
}

endjobs();

1;
