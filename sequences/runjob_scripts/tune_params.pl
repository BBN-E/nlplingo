#!/usr/bin/env perl
use strict;
use warnings;

# Standard libraries:
use FindBin;
use Getopt::Long;
use File::Basename;
use File::Copy;
use Data::Dumper;
use Cwd 'abs_path';

# Runjobs libraries:
use lib ("/d4m/ears/releases/Cube2/R2019_05_24_1/install-optimize$ENV{ARCH_SUFFIX}/perl_lib");
use lib( abs_path( dirname(__FILE__) ) . "/../../perl_lib" );
use runjobs4;
use File::PathConvert;
use File::Path 'make_path';
use Parameters;
use PerlUtils;
use RunJobsUtils;


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
my $LINUX_GPU_QUEUE = "allGPUs-k80";

my $SH = "/bin/bash";

my $PYTHON_CPU = $params->get("PYTHON_CPU");
my $PYTHON_GPU = $params->get("PYTHON_CPU");
my $TEXT_OPEN_PYTHON = $params->get("TEXT_OPEN");

my $nlplingo_mode = $params->get("nlplingo_mode");

my $NUM_RUNS = 1;

my $nlplingo_obj = Nlplingo->new(
    PYTHON_CPU => $PYTHON_CPU,
    PYTHON_GPU => $PYTHON_GPU,
    SERIF_PYTHON_PATH => $TEXT_OPEN_PYTHON,
    JOB_NAME => $exptName,
    LINUX_CPU_QUEUE => $LINUX_CPU_QUEUE,
    LINUX_GPU_QUEUE => $LINUX_GPU_QUEUE,
    USE_GPU => 1,
    SGE_VIRTUAL_FREE => ['32G','64G','192G']
);

my %params_to_search = (
    "positive_weight" => [1, 3, 5, 10],		# [1, 3, 5]
    "batch_size" => [20, 30, 50, 100],			# [20, 30, 50, 100]
    "num_epochs" => [10, 20, 30, 40, 50],			# [30, 40, 50, 60, 70]
    "neighbor_distance" => [0, 1],		# [0, 1]
    "hidden_layers" => ["256,256", "512,512", "768,768"],		# ["256,256", "512,512", "768,768"]
    "learning_rate" => [0.0001, 0.00001, 0.00033]	# [0.0001, 0.00001, 0.000001]
);

my %param_to_abbrev = (
    "positive_weight" => "w",
    "batch_size" => "b",
    "num_epochs" => "e",
    "neighbor_distance" => "n",
    "hidden_layers" => "hl",
    "learning_rate" => "lr",
    "number_of_feature_maps" => "f",
    "cnn_filter_length" => "fl",
    "position_embedding_vector_length" => "pemb",
    "entity_embedding_vector_length" => "eemb"
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
    add_serif_entity_mentions => $params->get("add_serif_entity_mentions"),
    average_embeddings => $params->get("average_embeddings"),
    entitymention_fuzzy_token_backing => $params->get("entitymention.fuzzy_token_backing"),
    anchor_fuzzy_token_backing => $params->get("anchor.fuzzy_token_backing")
);


my $search_argument_job_ids;
if($nlplingo_mode eq "train_argument") {
    $search_argument_job_ids =
        $nlplingo_obj->parameter_search_argument(
            dependant_job_ids => [],
            param_file => "tune_params.json",
            params_to_search => \%params_to_search,
            param_to_abbrev => \%param_to_abbrev,
            param_override => \%params_override,
            output_dir => "$expts/$exptName",
            serialized_list   => ""
        );
} elsif($nlplingo_mode eq "train_trigger_from_file") {
    $search_argument_job_ids =
        $nlplingo_obj->parameter_search_trigger(
            dependant_job_ids => [],
            param_file => "tune_params.json",
            params_to_search => \%params_to_search,
            param_to_abbrev => \%param_to_abbrev,
            param_override => \%params_override,
            output_dir => "$expts/$exptName",
            serialized_list   => ""
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
    add_serif_entity_mentions => $params->get("add_serif_entity_mentions"),
    average_embeddings => $params->get("average_embeddings"),
    entitymention_fuzzy_token_backing => $params->get("entitymention.fuzzy_token_backing"),
    anchor_fuzzy_token_backing => $params->get("anchor.fuzzy_token_backing")
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

