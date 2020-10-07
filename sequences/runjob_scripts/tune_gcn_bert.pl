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
use lib ("/d4m/better/releases/better/TMP2020_01_14.d92fc1a/perl_lib/");
use lib("/nfs/raid88/u10/users/jcai/nlplingo_make_release/perl_sequence/perl_lib");
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

our ($exp_root, $exp) = startjobs();
max_jobs(40);
my $expts = "$exp_root/expts";


######## These are the things user might want to change ########

my $LINUX_CPU_QUEUE = "allCPUs-sl610";
#my $LINUX_GPU_QUEUE = "allGPUs-sl610";
my $LINUX_GPU_QUEUE = "custom-pytorch-13";

my $SH = "/bin/bash";

my $PYTHON = $params->get("PYTHON");
my $GCN_PATH = $params->get("GCN_PATH");
my $TEXT_OPEN_PYTHON = $params->get("TEXT_OPEN") . ":" . $GCN_PATH;
my $NLPLINGO = $params->get("NLPLINGO");

#my $TRAIN_TEST_SCRIPT = "$NLPLINGO/nlplingo/tasks/train_test.py";
#my $AGGREGATE_SCRIPT = "/nfs/raid88/u10/users/ychan/nlplingo_models/bin/aggregate_scores.py";
#my $AVERAGE_SCRIPT = "/nfs/raid88/u10/users/ychan/nlplingo_models/bin/average_scores.py";


my $nlplingo_mode = $params->get("nlplingo_mode");

my $NUM_RUNS = 1;

my $nlplingo_obj = Nlplingo->new(
        PYTHON => $PYTHON,
        SERIF_PYTHON_PATH => $TEXT_OPEN_PYTHON,
        NLPLINGO_PATH => $NLPLINGO,
        JOB_NAME => $exptName,
        LINUX_CPU_QUEUE => $LINUX_CPU_QUEUE,
        LINUX_GPU_QUEUE => $LINUX_GPU_QUEUE,
        USE_GPU => 1,
        SGE_VIRTUAL_FREE => ['100G']
);

my %params_to_search = (
    "batch_size" => [30],
    "num_epochs" => [50],
    "dep_rel_dim" => [100, 200],
    "hidden_dim" => [300, 500],
    "learning_rate" => [0.0002, 0.0001, 0.00005, 0.00001],
    "num_graph_cnn_layers" => [2, 4, 6],
    "mlp_layers" => [2, 3, 4],
    "pytorch-optimizer" => ["Adam"],
    "prune_k" => [2, 3]
);


my %param_to_abbrev = (
    "batch_size" => "b",
    "num_epochs" => "e",
    "dep_rel_dim" => "drd", 
    "hidden_dim" => "hd",
    "learning_rate" => "lr",
    "num_graph_cnn_layers" => "gcnum",
    "mlp_layers" => "mlpnum",
    "pytorch-optimizer" => "pto",
    "prune_k" => "pk"
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
    add_serif_dep_graph => $params->get("add_serif_dep_graph"),
    add_serif_prop_adj => $params->get("add_serif_prop_adj")
);


my $search_argument_job_ids;
if($nlplingo_mode eq "train_argument") {
    $search_argument_job_ids =
        $nlplingo_obj->parameter_search_argument(
            dependant_job_ids => [],
            param_file => "tune_params_gcn_bert.json",
            params_to_search => \%params_to_search,
            param_to_abbrev => \%param_to_abbrev,
            param_override => \%params_override,
            output_dir => "$expts/$exptName",
            serialized_list => ""
        );
} elsif($nlplingo_mode eq "train_trigger_from_file") {
    $search_argument_job_ids =
        $nlplingo_obj->parameter_search_trigger(
            dependant_job_ids => [],
            param_file => "tune_params_gcn_bert.json",
            params_to_search => \%params_to_search,
            param_to_abbrev => \%param_to_abbrev,
            param_override => \%params_override,
            output_dir => "$expts/$exptName",
            serialized_list => ""
        );
} else{
    die "nlplingo_mode unsupported. " .
        "Use {train_trigger_from_file|train_argument}\n";
}

# These could be different but aren't for right now
my %base_params_override = (
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

my $find_best_params_file_job_ids = $nlplingo_obj->find_best_params_file(
    dependant_job_ids => $search_argument_job_ids,
    base_params_file => "base_tune_params.json",
    base_param_override => \%base_params_override,
    instance_name => $nlplingo_mode,
    input_dir => "$expts/$exptName",
    output_dir => "$expts/$exptName"
);

print("When completed look in $expts/$exptName/best.params.json");


endjobs();


