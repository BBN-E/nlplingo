
package main;

use strict;
use warnings;

# Standard libraries:
use Getopt::Long;
use File::Basename;
use Data::Dumper;

# Runjobs libraries:
use lib "/d4m/ears/releases/Cube2/R2016_07_21/install-optimize-x86_64/perl_lib";
use runjobs4;
use File::PathConvert;
use File::Basename;
use Parameters;
use PerlUtils;
use RunJobsUtils;

my $paramFile = $ARGV[0];
print "Got param file $paramFile\n";
my $params = Parameters::loadAndPrint($paramFile);
defined($params) or die "Could not load params";

our ($exp_root, $exp) = startjobs();

my $exptName = $params->get('exptName');
my $expts = "$exp_root/expts";

max_jobs(100,);

my $baseDir = "$expts/$exptName";

###################
# End of Template #
###################


############################
# Start of input variables #
############################

my $LINUX_QUEUE = "nongale-sl6";
my $SINGULARITY_GPU_QUEUE = "allGPUs-sl69-non-k10s";
my $GPU_QUEUE = "nongaleGPUs";

#### variables
my $set_size = 10;
my $docidList = $params->get("inputfile.list");
my $maximum_allowed_bert_tokens_per_sentence = $params->get('maximum_allowed_bert_tokens_per_sentence');
my $bert_layers = $params->get('bert_layers');

#### python and scripts
my $python_path = $params->get("python.path");	# path to: NLPLingo repo, SERIF python, BERT repo
my $python_bin = $params->get("python.bin");
my $do_bert_tokenization_bin = $params->get("do_bert_tokenization.bin");
my $do_npz_embeddings_bin = $params->get("do_npz_embeddings.bin");
my $add_npz_suffix_bin = $params->get("add_npz_suffix.bin");

#### bert model
my $bert_repo = $params->get("bert.repo");
my $bert_model = $params->get("bert.model");
my $vocab_file = "$bert_model/vocab.txt";

#### relative output directories
my $outdir_tokens = "$baseDir/tokens";
my $outdir_embeddings = "$baseDir/embeddings";
my $outdir_npz = "$baseDir/npz_embeddings";


############################
# Preparation: create dirs #
# ##########################

my $listdir = $baseDir . "/lists";
if(!-e $listdir) {
  my $cmd = "mkdir -p $listdir";
  `$cmd`;
}

if(!-e $outdir_tokens) {
  my $cmd = "mkdir -p $outdir_tokens";
  `$cmd`;
}

if(!-e $outdir_embeddings) {
  my $cmd = "mkdir -p $outdir_embeddings";
  `$cmd`;
}

if(!-e $outdir_npz) {
  my $cmd = "mkdir -p $outdir_npz";
  `$cmd`;
}


#########################################################
# Preparation: divide input into sub-lists #
# #######################################################

my @docids = ();
open IN, $docidList;
while(<IN>) {
  chomp;
  my $line = $_;
  push(@docids, $line); 
}
close IN;

#my $set_size = int(scalar(@docids)/200 + 0.5);
my $i = 0;
my $set_number = 0;
while($i<=$#docids) {
  my @ids = ();
  my $j = 0;
  for($j=$i; $j<=$#docids && $j<($i+$set_size); $j++) {
    push(@ids, $docids[$j]);
  }  
  $i = $j;

  my $listFile = "$baseDir/lists/set$set_number";
  open OUT, ">$listFile";
  for(my $j=0; $j<=$#ids; $j++) {
    print OUT $ids[$j] . "\n";  
  }
  close OUT;

  my $cmd = "mkdir -p $outdir_tokens/set$set_number";
  `$cmd`;

  #my $countFile = "$baseDir/counts/set$set_number.count";

  #my $cmd = "python $bin --filelist $listFile --outfile $countFile";
  #my $job = runjobs4::runjobs([], "$exptName/counts_set$set_number", { BATCH_QUEUE => $params->get("batchQueue"), SGE_VIRTUAL_FREE => "2G" }, ["$cmd"]);

  $set_number += 1;
}


##################################################
# Do the actual runjobs of producing BERT tokens #
# ################################################

my @bert_token_jobs = ();
my @list_files = glob($baseDir."/lists/*");
for($i=0; $i<=$#list_files; $i++) {
  my $filepath = $list_files[$i];
  my $filename = basename($filepath);

  #my $cmd = "PYTHONPATH=$python_path $python_bin --filelist $filepath --bert_vocabfile $vocab_file --outdir $outdir";
  #my $job = runjobs4::runjobs([], "$exptName/bert_tokens/$filename", { BATCH_QUEUE => $LINUX_QUEUE, SGE_VIRTUAL_FREE => "2G" }, ["$cmd"]);

  my $job_id = runjobs4::runjobs(
        [],
        "$exptName/bert_tokens/$filename",
        {
            BATCH_QUEUE      => $GPU_QUEUE,
            #SGE_VIRTUAL_FREE => "2G",
            #QUEUE_PRIO       => $QUEUE_PRIO,
            python_path      => $python_path,
            command          =>  "$python_bin $do_bert_tokenization_bin",
            args             => "--filelist $filepath --bert_vocabfile $vocab_file --outdir $outdir_tokens/$filename --maximum_allowed_bert_tokens_per_sentence $maximum_allowed_bert_tokens_per_sentence"
        },
        ["/bin/bash", "run_python.sh"]
  );
  push(@bert_token_jobs, $job_id);
}

# cat ../../../repos/nlp_sandbox/expts/bert_ace/lists/set0  | awk -F/ '{print $NF}'


######################################################
# Do the actual runjobs of producing BERT embeddings #
# ####################################################

my @bert_embedding_jobs = ();
@list_files = glob($baseDir."/lists/*");
for($i=0; $i<=$#list_files; $i++) {
  my $filepath = $list_files[$i];
  my $filename = basename($filepath);

  my $job_id = runjobs4::runjobs(
    \@bert_token_jobs,
    "$exptName/bert_embeddings/$filename",
    {
      BATCH_QUEUE    => $GPU_QUEUE,
      SGE_VIRTUAL_FREE => "8G",
      filelist       => $filepath,
      bert_repo      => $bert_repo,
      python         => $python_bin,
      indir          => "$outdir_tokens/$filename",
      outdir         => $outdir_embeddings,
      bert_model     => $bert_model,
      bert_layers    => $bert_layers
    },
    ["/bin/bash", "run_bert_embeddings.sh"]
  );
  push(@bert_embedding_jobs, $job_id);
}


############################
# Do npz on the embeddings #
# ##########################

@list_files = glob($baseDir."/lists/*");
my @npz_jobs = ();
for($i=0; $i<=$#list_files; $i++) {
  my $filepath = $list_files[$i];
  my $filename = basename($filepath);

  my $job_id = runjobs4::runjobs(
        \@bert_embedding_jobs,
        "$exptName/npz_embeddings/$filename",
        {
            BATCH_QUEUE      => $LINUX_QUEUE,
            SGE_VIRTUAL_FREE => "8G",
            #QUEUE_PRIO       => $QUEUE_PRIO,
            python_path      => $python_path,
            command          =>  "$python_bin $do_npz_embeddings_bin",
            args             => "--embeddings_dir $outdir_embeddings --token_dir $outdir_tokens/$filename --token_map_dir $outdir_tokens/$filename --outdir $outdir_npz"
        },
        ["/bin/bash", "run_python.sh"]
  );
  push(@npz_jobs, $job_id);
}


runjobs4::runjobs(
        \@npz_jobs,
        "$exptName/add_npz_suffix",
        {
            BATCH_QUEUE      => $LINUX_QUEUE,
            SGE_VIRTUAL_FREE => "2G",
            python_path      => $python_path,
            command          =>  "$python_bin $add_npz_suffix_bin",
            args             => "$docidList $outdir_npz $baseDir/serif_embedding.filelist"
        },
        ["/bin/bash", "run_python.sh"]
);


endjobs();



