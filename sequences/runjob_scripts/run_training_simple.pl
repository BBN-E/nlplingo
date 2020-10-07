#!/bin/env perl

use strict;
use warnings;

# This should be used at first, due to we want to use a new enough runjobs4
use FindBin qw($Bin $Script);
use Cwd;

use File::Basename;
use File::Path;
use File::Copy;

package main;

my $textopen_root;
my $hume_repo_root;
my $learnit_root;
my $nlplingo_root;

BEGIN{
    $textopen_root = "/d4m/nlp/releases/text-open/R2020_04_15";
    $hume_repo_root = "/d4m/nlp/releases/Hume/R2020_04_15_1";
    $learnit_root = "/d4m/nlp/releases/learnit/R2020_04_14";
    $nlplingo_root = "/d4m/nlp/releases/nlplingo/R2020_04_15";
    unshift(@INC, "/d4m/ears/releases/runjobs4/R2019_03_29/lib");
    unshift(@INC, "$textopen_root/src/perl/text_open/lib");
    unshift(@INC, "$learnit_root/lib/perl_lib/");
}

# use lib "/d4m/ears/releases/runjobs4/R2019_07_16/lib";
use runjobs4;
use Utils;
use File::Path qw(make_path);

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

# prep runjobs


my $LINUX_GPU_QUEUE = "custom-pytorch-13";
my $QUEUE_PRIO = '5'; # Default queue priority
my ($exp_root, $exp) = startjobs("batch_queue" => $LINUX_GPU_QUEUE,
                                "queue_priority" => $QUEUE_PRIO,
                                "queue_mem_limit" => '100G',
                                "max_memory_over" => '4G'
                                );

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
}

my $JOB_NAME = get_param($params, "job_name");
my $dataset = get_param($params, "dataset");
my $encoder = get_param($params, "encoder");
my $mode = get_param($params, "mode");
my $model_config = get_param($params, "model_config");
my $epoch = get_param($params, "epoch");
my $job_cores = get_param($params, "job_cores");

sub getLoggingTime {

    my ($sec,$min,$hour,$mday,$mon,$year,$wday,$yday,$isdst)=localtime(time);
    my $nice_timestamp = sprintf ( "%04d%02d%02d_%02d_%02d_%02d",
                                   $year+1900,$mon+1,$mday,$hour,$min,$sec);
    return $nice_timestamp;
}
my $ts = getLoggingTime();

# Create relevant directories
(my $expbase, undef) = Utils::make_output_dir("$exp_root/expts/$JOB_NAME" . "_" . $ts, "$JOB_NAME/mkdir_job_directory" . "_" . $ts, []);
my $output_dir="$expbase"; 

# Python code directory
# my $bin_dir = "$exp_dir/bin/";
# my $SINGULARITY_WRAPPER = "$bin_dir/relations/scripts/run-in-singularity-container.sh";
# my $bash_file = "$bin_dir/relations/scripts/train-runjobs.sh";
# my $model_config = "/nfs/raid88/u10/users/jcai/code/nlplingo_test_scripts/scripts/params/eer_full.json"

my @train_jobs = ();

# Change $exp_id parameter if you desire to alter the output folder where trained model and statistics are emitted
my $exp_id = "bert_family_train_$ts";

push @train_jobs, runjobs([], "$exp_id",
    {
        SGE_VIRTUAL_FREE => "80G",
        ATOMIC => 1,
        job_cores => $job_cores,
        output_dir => $output_dir,
        dataset => $dataset,
        encoder => $encoder,
        epoch => $epoch
    },
    ["cp ", "$model_config", " $output_dir"],
    ["/d4m/material/software/python/singularity/bin/singularity-python.sh -i python3.6-cuda10.0  -l '/nfs/raid88/u10/users/jcai/code/fresh/nlplingo:/nfs/raid88/u10/users/jcai/modern/text-open/src/python' -v '/nfs/raid87/u11/users/hqiu/miniconda_prod/envs/nlplingo-gpu/' --gpu /nfs/raid88/u10/users/jcai/code/fresh/nlplingo/nlplingo/tasks/train_test.py --params ", "$model_config", " --mode $mode"]);

endjobs();
