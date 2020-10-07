#!/usr/bin/env perl
use strict;
use warnings;

# Standard libraries:
use FindBin;
use Getopt::Long;
use File::Basename;
use File::Copy;
use Data::Dumper;

# Runjobs libraries:
use lib ("/d4m/ears/releases/Cube2/R2019_05_24_1/install-optimize$ENV{ARCH_SUFFIX}/perl_lib");
use lib ("/nfs/raid88/u10/users/jcai/code/better/perl_lib");
# use lib("/nfs/raid84/u12/jfaschin/better_staging/src/git/nlplingo/develop/nlplingo_make_release/perl_lib");
use runjobs4;
use File::PathConvert;
use File::Path 'make_path';
use Parameters;
use PerlUtils;
use RunJobsUtils;
use Cwd 'abs_path';

use Utils;
use Nlplingo;

# Package declaration:
package main;


my $paramFile = $ARGV[0];
print "Got param file $paramFile\n";
my $params = Parameters::loadAndPrint($paramFile);
defined($params) or die "Could not load params";

my $exptName = $params->get('exptName');

our ($exp_root, $exp) = startjobs("queue_priority" => 5);
max_jobs(40);
my $expts = "$exp_root/expts";


######## These are the things user might want to change ########

my $LINUX_CPU_QUEUE = "allCPUs-sl610";
my $LINUX_GPU_QUEUE = "custom-pytorch-13";

my $SH = "/bin/bash";

my $PYTHON_CPU = $params->get("PYTHON_CPU");
my $PYTHON_GPU = $params->get("PYTHON_GPU");
my $TEXT_OPEN_PYTHON = $params->get("TEXT_OPEN");
my $NLPLINGO = $params->get("NLPLINGO");

#my $TRAIN_TEST_SCRIPT = "$NLPLINGO/nlplingo/tasks/train_test.py";
#my $AGGREGATE_SCRIPT = "/nfs/raid88/u10/users/ychan/nlplingo_models/bin/aggregate_scores.py";
#my $AVERAGE_SCRIPT = "/nfs/raid88/u10/users/ychan/nlplingo_models/bin/average_scores.py";


my $nlplingo_mode = $params->get("nlplingo_mode");

my $NUM_RUNS = 1;

my $nlplingo_obj = Nlplingo->new(
        PYTHON_CPU => $PYTHON_CPU,
        PYTHON_GPU => $PYTHON_GPU,
        SERIF_PYTHON_PATH => $TEXT_OPEN_PYTHON,
        NLPLINGO_PATH => $NLPLINGO,
        JOB_NAME => $exptName,
        LINUX_CPU_QUEUE => $LINUX_CPU_QUEUE,
        LINUX_GPU_QUEUE => $LINUX_GPU_QUEUE,
        USE_GPU => 1,
        SGE_VIRTUAL_FREE => ['150G']
);

my %params_to_search = (
    "num_epochs" => [300]
);


my %param_to_abbrev = (
    "num_epochs" => "ne"
);


my %params_override = (
    train_file_list => $params->get("train_filelist"),
    dev_file_list => $params->get("dev_filelist"),
    test_file_list => $params->get("test_filelist"),
    domain_ontology => $params->get("domain_ontology"),
    features => $params->get("features"),
    model_type => $params->get("model_type"),
    negative_trigger_words => $params->get("negative_trigger_words"),
    vector_size => $params->get("vector_size"),
    add_serif_event_mentions => $params->get("add_serif_event_mentions"),
    add_serif_entity_mentions => $params->get("add_serif_entity_mentions")
);


my $search_argument_job_ids;
if($nlplingo_mode eq "train_evevr_from_file") {
    $search_argument_job_ids =
        $nlplingo_obj->parameter_search_evevr(
            dependant_job_ids => [],
            param_file => "eer_full.json",
            params_to_search => \%params_to_search,
            param_to_abbrev => \%param_to_abbrev,
            param_override => \%params_override,
            output_dir => "$expts/$exptName",
        );
} else{
    die "nlplingo_mode unsupported. " .
        "Use {train_trigger_from_file|train_argument}\n";
}

endjobs();
