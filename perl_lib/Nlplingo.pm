package Nlplingo;
use strict;
use warnings;

use Carp;

use lib ("/d4m/ears/releases/Cube2/R2019_05_24_1/install-optimize$ENV{ARCH_SUFFIX}/perl_lib");
use runjobs4;
use File::Basename;
use Cwd 'abs_path';
use Utils 'cartesian';
#require Exporter;

#our @ISA = qw(Exporter);
#our @EXPORT = qw(here setup_params );  # don't export method names!


sub new{
    my ($class, %rest) = @_;
    my $self;

    if ( %rest ) {
        $self = \%rest;
    } else {
        $self = {};
    }

    bless $self, $class;

    my @fs = $self->fields();
  
    # Check the passed fields
    foreach my $k ( keys %{ $self } )
    {
        croak "In new(), $class doesn't have a $k field but was given it as a parameter" unless grep( /^$k$/, @fs );
    }

    my @missing = grep { ! defined($self->{$_}) } @fs;
    if ( @missing ) {
        croak "In new(), $class not passed mandatory field(s): @missing\n";
    }

    $self->init();

    return $self;
}

sub init{
    my $self = shift;
    $self->{MODULE_NAME} = "Nlplingo";
    $self->{NLPLINGO_PATH} = abs_path( dirname(__FILE__) . "/../" );
    $self->{PYTHONPATH} = $self->{SERIF_PYTHON_PATH} . ":" . $self->{NLPLINGO_PATH};
    $self->{TRAIN_TEST_SCRIPT} = $self->{NLPLINGO_PATH} . "/nlplingo/tasks/train_test.py";	# default
    $self->{MODIFY_PARAMS_SCRIPT} = 
        $self->{NLPLINGO_PATH} . 
        "/perl_lib/scripts/modify_params.py";
    $self->{FIND_BEST_PARAMS_FILE_SCRIPT} =
        $self->{NLPLINGO_PATH} .
        "/perl_lib/scripts/find_best_params_file.py";
    $self->{AVERAGE_SCORE_FILES_SCRIPT} =
        $self->{NLPLINGO_PATH} .
        "/perl_lib/scripts/average_scores.py";
}

sub fields{
    return (
        'PYTHON_CPU',
        'PYTHON_GPU',
        'SERIF_PYTHON_PATH',
        'JOB_NAME',
        'LINUX_CPU_QUEUE',
        'LINUX_GPU_QUEUE',
        'USE_GPU',
        'SGE_VIRTUAL_FREE',
    )
}

sub subroutine_name{
    # Since a user of this subroutine is one call frame above
    # EXPR = 1
    my $sub_name = (caller(1))[3];
    # Find replace on : to make both SGE and linux filenames happy
    $sub_name =~ s/:/_/g;
    return $sub_name;
}


=pod
=item train_command()

This subroutine returns a command using NLPLingo's train_test script that is
ready to be used in a RunJobs call.  Either a parameter template or a completed
parameter json must be supplied.

Inputs:
    params_template - Path to params file template.
    params_json     - Path to completed params file.
    mode            - Training mode.
    serialized      - Path to serialized NPZ list.  Optional.
Outputs:
    An arrayref of command strings ready to be used in RunJobs.

=cut
sub train_command{
    my ($self, %args) = @_;
    my $params_template = $args{params_template};
    my $params_json = $args{params_json};
    my $mode = $args{mode};
    my $serialized = $args{serialized};

    my $PYTHON = $self->{PYTHON_CPU};
    if ($self->{USE_GPU}) {
        $PYTHON = $self->{PYTHON_GPU};
    }

    my $serialized_opt = "";
    if (defined $serialized && $serialized ne "") {
        $serialized_opt = "--serialize_list $serialized";
    }

    my $command = (
        "env PYTHONPATH=$self->{PYTHONPATH} "
            . "KERAS_BACKEND=tensorflow "
            . "$PYTHON "
            . "$self->{TRAIN_TEST_SCRIPT} "
            . "--mode $mode "
            . "$serialized_opt "
            . "--params "
    );

    my $ret;
    if (defined $params_template) {
        if (defined $params_json) {
            die "Both a params template and completed json supplied, stopping";
        } else {
            $ret = [ $command, $params_template ]
        }
    } else {
        if (!defined $params_json) {
            die "Neither a params template nor a completed json supplied, stopping";
        } else {
            $ret = [ "$command $params_json" ]
        }
    }

    return $ret;
}

sub parameter_search{
    my ($self, %args) = @_;
    my $sub_name = subroutine_name();
    my $dependant_job_ids = $args{dependant_job_ids};
    my $param_file = $args{param_file};
    my %params_to_search = %{$args{params_to_search}};
    my %param_to_abbrev = %{$args{param_to_abbrev}};
    my %param_override = %{$args{param_override}};
    my $output_dir = $args{output_dir};
    my $model_file = $args{model_file}; 
    my $nlplingo_mode = $args{nlplingo_mode};
    my $serialized_list = $args{serialized_list};
    my $serialized_k_folds = $args{serialized_k_folds};  # TODO include for unserialized runs?

    my @names = sort(keys(%params_to_search));

    my @params_cartesian_product = Utils::cartesian(@params_to_search{reverse(@names)});

    my @nlplingo_jobs = ();

    foreach(@params_cartesian_product){

        my @combination = @{$_};
        my $QUEUE;
        my $processing_type;
        my $PYTHON;
        if($self->{USE_GPU}){
            $QUEUE = $self->{LINUX_GPU_QUEUE};
            $processing_type = "gpu";
            $PYTHON = $self->{PYTHON_GPU};
        }else{
            $QUEUE = $self->{LINUX_CPU_QUEUE};
            $processing_type = "cpu";
            $PYTHON = $self->{PYTHON_CPU};
        }
        my %run_params = ( 
            BATCH_QUEUE =>  $QUEUE,
            SGE_VIRTUAL_FREE => $self->{SGE_VIRTUAL_FREE}
        );
        my $combination_name = "";
        foreach(map [ $_, $combination[$_] ], 0 .. $#combination){
            my $param_name = $names[ @{$_}[0] ];
            my $param_value = @{$_}[1];
            $run_params{ $param_name } = $param_value;
            $combination_name = $combination_name . $param_to_abbrev{ $param_name } . $param_value; 
        }
        $combination_name =~ tr/,/_/;
        my $combo_output_dir = "$output_dir/$combination_name";

        $run_params{model_file} = $model_file;
        $run_params{output_dir} = $combo_output_dir;
        while ((my $k, my $v) = each %run_params) {
            print "$k => $v\n";
        }
        while ((my $k, my $v) = each %param_override) {
            $run_params{$k} = $v;
            print "Override: $k => $v\n";
        }
        my $job_name = $processing_type . "/". # FIXME Why does this come before JOB_NAME?
            $self->{JOB_NAME} . "/" .  
            $sub_name . "/" . 
            $nlplingo_mode . "/". 
            $combination_name;

        my $serialized_list_opt;
        if (!defined $serialized_list || $serialized_list eq "") {
            $serialized_list_opt = "";
        } else {
            $serialized_list_opt = "--serialize_list $serialized_list ";
        }

        if (!defined $serialized_k_folds || $serialized_k_folds eq "") {
            my $job = runjobs(
                $dependant_job_ids,
                $job_name,
                \%run_params,
                [ "mkdir -p $combo_output_dir" ],
                train_command(
                    params_template => $param_file,
                    mode            => $nlplingo_mode,
                    serialized      => $serialized_list,
                ),
                [ "cp -t $combo_output_dir ", $param_file ], # There ought to be a better way to store etemplates in expts
                [ "bash -c \'for i in $combo_output_dir/*\$(basename " . $param_file . "); do mv -f \$i $combo_output_dir/params.json; done\'"]
            );
            push @nlplingo_jobs, $job
        } else {
            my @fold_jobs = ();
            foreach my $fold (0..($serialized_k_folds - 1)) {

                my $fold_job_name = $job_name . "/fold$fold";
                my $fold_output_dir = "$combo_output_dir/fold$fold";
                $run_params{output_dir} = $fold_output_dir;

                $serialized_list_opt = $serialized_list_opt .
                    "--k_partitions $serialized_k_folds " .
                    "--partition_id $fold ";


                my $job = runjobs(
                    $dependant_job_ids,
                    $fold_job_name,
                    \%run_params,
                    [ "mkdir -p $fold_output_dir" ],
                    # Can't replace yet.  TODO add CV opts to train_command()
                    [
                        "env PYTHONPATH=$self->{PYTHONPATH} KERAS_BACKEND=tensorflow " .
                            "$PYTHON " .
                            "$self->{TRAIN_TEST_SCRIPT} ".
                            "--mode $nlplingo_mode " .
                            "$serialized_list_opt " .
                            "--params",
                        $param_file
                    ],
                    [ "cp -t $fold_output_dir ", $param_file ], # There ought to be a better way to store etemplates in expts
                    [ "bash -c \'for i in $fold_output_dir/*\$(basename " . $param_file . "); do mv -f \$i $fold_output_dir/params.json; done\'"]
                );
                push @fold_jobs, $job;
            }
            # Micro-average scores across all folds, and place separately
            # (so the originals aren't picked up by find_best_params_file)

            my $test_filelist = "$combo_output_dir/test.score.list";
            my $test_score_gather_job_ids = Utils::gather_docs_to_list(
                dependant_job_ids => \@fold_jobs,
                job_name => $self->{JOB_NAME},
                instance_name => "$nlplingo_mode.$combination_name.test_score_list",
                gather_folder => $combo_output_dir,
                list_file => $test_filelist,
                search_string => "test.score",
                linux_cpu_queue=> $self->{LINUX_CPU_QUEUE},
            );

            my $test_score_file = "$combo_output_dir/aggregated.test.score";
            my $average_job_name = "cpu/$self->{JOB_NAME}/$sub_name/$nlplingo_mode/$combination_name/average_scores";
            my $average_scores_job = runjobs(
                $test_score_gather_job_ids,
                $average_job_name,
                {
                    BATCH_QUEUE      => $self->{LINUX_CPU_QUEUE},
                    SGE_VIRTUAL_FREE => "1G"
                },
                [ "$self->{PYTHON_CPU} " .
                    "$self->{AVERAGE_SCORE_FILES_SCRIPT} " .
                    "--input_filelist $test_filelist " .
                    "--output_file $test_score_file" ]
            );
            push @nlplingo_jobs, $average_scores_job;
        }
    }
    return \@nlplingo_jobs
}

sub parameter_search_argument{
    my ($self, %args) = @_;

    my $model_file = $args{output_dir};
    if(exists($args{model_file})) {
        $model_file = $args{model_file};
    }
    return $self->parameter_search(
        dependant_job_ids => $args{dependant_job_ids},
        param_file => $args{param_file},
        params_to_search => $args{params_to_search},
        param_to_abbrev => $args{param_to_abbrev},
        param_override => $args{param_override},
        output_dir => $args{output_dir},
        model_file => $model_file,
        serialized_list => $args{serialized_list},
        serialized_k_folds => $args{serialized_k_folds},  # TODO include for unserialized runs?
        nlplingo_mode => "train_argument"
    );
}

sub parameter_search_trigger{
    my ($self, %args) = @_;

    my $model_file = $args{output_dir};
    if(exists($args{model_file})) {
        $model_file = $args{model_file};
    }
    return $self->parameter_search(
        dependant_job_ids => $args{dependant_job_ids},
        param_file => $args{param_file},
        params_to_search => $args{params_to_search},
        param_to_abbrev => $args{param_to_abbrev},
        param_override => $args{param_override},
        output_dir => $args{output_dir},
        model_file => $model_file,
        serialized_list => $args{serialized_list},
        serialized_k_folds => $args{serialized_k_folds},  # TODO include for unserialized runs?
        nlplingo_mode => "train_trigger_from_file"
    );
}

sub parameter_search_sequence{
    my ($self, %args) = @_;

    my $model_file = $args{output_dir};
    if(exists($args{model_file})) {
        $model_file = $args{model_file};
    }
    return $self->parameter_search(
        dependant_job_ids => $args{dependant_job_ids},
        param_file => $args{param_file},
        params_to_search => $args{params_to_search},
        param_to_abbrev => $args{param_to_abbrev},
        param_override => $args{param_override},
        output_dir => $args{output_dir},
        model_file => $model_file,
        serialized_list => $args{serialized_list},
        serialized_k_folds => $args{serialized_k_folds},  # TODO include for unserialized runs?
        nlplingo_mode => "train_ner"
    );
}

=pod
=item find_best_params_file()

This subroutine finds the best nlplingo parameters for a nlplingo parameter search directory
and reports back a parameter file with the best parameters.

Inputs:
    dependant_job_ids  - Arrayref to dependant runjobs
    base_params_file   - Path to params file with no extractors
    base_param_override - Hashref of parameters to override for 
                          the base_params_file 
    instance_name      - Unique name for the instance
    input_dir          - Path to directory containing test score files
    output_dir         - Path to directory where the output files go
Outputs:
    job_ids          - Arrayref to last ran job_ids
    best_params_file - Path to best params file selected by subroutine

=cut
sub find_best_params_file{
    my ($self, %args) = @_;
    my $sub_name = subroutine_name();
    my $dependant_job_ids = $args{dependant_job_ids};
    my $base_params_file = $args{base_params_file};
    my %base_param_override = %{$args{base_param_override}};
    my $instance_name = $args{instance_name};
    my $input_dir = $args{input_dir};
    my $output_dir = $args{output_dir};
    my $aggregated = $args{aggregated};

    my $search_string = "test.score";
    if ($aggregated) {
        $search_string = "aggregated.$search_string";
    }

    my $test_score_list = "$output_dir/$instance_name.test.score.list";
    my $test_score_gather_job_ids = Utils::gather_docs_to_list(
        dependant_job_ids => $dependant_job_ids,
        instance_name => $instance_name . "_test_score_list",
        gather_folder => $input_dir,
        list_file => $test_score_list,
        job_name => $self->{JOB_NAME},
        search_string => $search_string,
        linux_cpu_queue=> $self->{LINUX_CPU_QUEUE},
    );

    my $best_instance_params_file = "$output_dir/best.$instance_name.params.json";
    my $best_params_file = "$output_dir/best.params.json";
    my %run_params = (
        BATCH_QUEUE => $self->{LINUX_CPU_QUEUE},
        output_dir => $output_dir,
    );
    while ((my $k, my $v) = each %base_param_override) {
        $run_params{$k} = $v;
    }

    # in some cases it's less complicated to use existing parameters as the base
    my $modify_rj_cmd;
    my $modify_cmd = ("env PYTHONPATH=$self->{PYTHONPATH} " .
         "$self->{PYTHON_CPU} " .
         "$self->{MODIFY_PARAMS_SCRIPT} " .
         "--trigger_params_file $best_instance_params_file " .
         "--output_params_file $best_params_file ");
    if (defined $base_params_file) {
        $modify_cmd = $modify_cmd . "--base_params_file ";
        $modify_rj_cmd = [ $modify_cmd, $base_params_file ];
    } else {
        $modify_rj_cmd = [ $modify_cmd ];
    }

    my $find_best_params_file_job_id = runjobs(
        $test_score_gather_job_ids,
        "$self->{JOB_NAME}/$sub_name", 
        \%run_params,
        [
         "env PYTHONPATH=$self->{PYTHONPATH} " . 
         "$self->{PYTHON_CPU} " . 
         "$self->{FIND_BEST_PARAMS_FILE_SCRIPT} " . 
         "--input_score_list $test_score_list " .
         "--output_params_file $best_instance_params_file" 
        ],
        $modify_rj_cmd,
    );
    return (
        [$find_best_params_file_job_id],
        $best_params_file
    );

}

=pod
=item find_best_params_file_trigger_argument()

This subroutine finds the best nlplingo parameters for triggers and arguments
and reports back a parameter file with the best parameters for both.

Inputs:
    dependant_job_ids   - Arrayref to dependant runjobs
    base_params_file    - Path to params file with no extractors
    base_param_override - Hashref of parameters to override for 
                          the base_params_file 
    trigger_input_dir   - Path to directory containing trigger dev score files
    argument_input_dir  - Path to directory containing argument dev score files
    output_dir          - Path to directory where the output files go
Outputs:
    job_ids          - Arrayref to last ran job_ids
    best_params_file - Path to best params file selected by subroutine

=cut
sub find_best_params_file_trigger_argument{
    my ($self, %args) = @_;
    my $sub_name = subroutine_name();
    my $dependant_job_ids = $args{dependant_job_ids};
    my $base_params_file = $args{base_params_file};
    my %base_param_override = %{$args{base_param_override}};
    my $trigger_input_dir = $args{trigger_input_dir};
    my $argument_input_dir = $args{argument_input_dir};
    my $output_dir = $args{output_dir};

    my @gather_job_ids = ();
    my $trigger_dev_score_list = "$output_dir/trigger.dev.score.list";
    my $trigger_dev_score_gather_job_ids = Utils::gather_docs_to_list(
        dependant_job_ids => $dependant_job_ids,
        instance_name => "trigger_dev_score_list",
        gather_folder => $trigger_input_dir,
        list_file => $trigger_dev_score_list,
        job_name => $self->{JOB_NAME},
        search_string => "dev.score",
        linux_cpu_queue=> $self->{LINUX_CPU_QUEUE},
    );
    push @gather_job_ids, @{$trigger_dev_score_gather_job_ids};
    my $argument_dev_score_list = "$output_dir/argument.dev.score.list";    
    my $argument_dev_score_gather_job_ids = Utils::gather_docs_to_list(
        dependant_job_ids => $dependant_job_ids,
        instance_name => "argument_dev_score_list",
        gather_folder => $argument_input_dir,
        list_file => $argument_dev_score_list,
        job_name => $self->{JOB_NAME},
        search_string => "dev.score",
        linux_cpu_queue=> $self->{LINUX_CPU_QUEUE},
    );
    push @gather_job_ids, @{$argument_dev_score_gather_job_ids};
    my $best_trigger_params_file = "$output_dir/best.trigger.params.json";
    my $best_argument_params_file = "$output_dir/best.argument.params.json";
    my $best_params_file = "$output_dir/best.params.json";
    my %run_params = (
        BATCH_QUEUE => $self->{LINUX_CPU_QUEUE},
        output_dir => $output_dir,
    );
    while ((my $k, my $v) = each %base_param_override) {
        $run_params{$k} = $v;
    }
    my $find_best_params_file_job_id = runjobs(
        \@gather_job_ids, 
        "cpu/$self->{JOB_NAME}/$sub_name", 
        \%run_params,
        [
         "env PYTHONPATH=$self->{PYTHONPATH} " . 
         "$self->{PYTHON_CPU} " . 
         "$self->{FIND_BEST_PARAMS_FILE_SCRIPT} " . 
         "--input_score_list $trigger_dev_score_list " . 
         "--output_params_file $best_trigger_params_file" 
        ],
        [
         "env PYTHONPATH=$self->{PYTHONPATH} " . 
         "$self->{PYTHON_CPU} " . 
         "$self->{FIND_BEST_PARAMS_FILE_SCRIPT} " . 
         "--input_score_list $argument_dev_score_list " . 
         "--output_params_file $best_argument_params_file" 
        ],
        [
         "env PYTHONPATH=$self->{PYTHONPATH} " . 
         "$self->{PYTHON_CPU} " . 
         "$self->{MODIFY_PARAMS_SCRIPT} " . 
         "--trigger_params_file $best_trigger_params_file " .
         "--argument_params_file $best_argument_params_file " .
         "--output_params_file $best_params_file ",
         "$base_params_file"
        ]
    );
    return (
        [$find_best_params_file_job_id],
        $best_params_file
    );

}

sub decode{
    my ($self, %args) = @_;
    my $sub_name = subroutine_name();
    my $dependant_job_ids = $args{dependant_job_ids};
    my $instance_name     = $args{instance_name};
    my %param_override    = %{$args{param_override}}; # runjobs params, 
                                                    # not nlplingo params
    my $param_file        = $args{param_file};
    my $output_dir        = $args{output_dir};
    my $mode              = $args{mode};

    my $QUEUE;
    my $processing_type; # TODO: this should be done in the constructor
    my $PYTHON;

    if($self->{USE_GPU}){
        $QUEUE = $self->{LINUX_GPU_QUEUE};
        $processing_type = "gpu";
        $PYTHON = $self->{PYTHON_GPU};
    }else{
        $QUEUE = $self->{LINUX_CPU_QUEUE};
        $processing_type = "cpu";
        $PYTHON = $self->{PYTHON_CPU};
    }
    my $nlplingo_mode = $mode;
    my $job_name = $processing_type . "/". 
        $self->{JOB_NAME} . "/" .   
        $sub_name . "/" . 
        $nlplingo_mode . "/". 
        $instance_name;

    my %run_params = ( 
        BATCH_QUEUE =>  $QUEUE,
        SGE_VIRTUAL_FREE => $self->{SGE_VIRTUAL_FREE}
    );

    while ((my $k, my $v) = each %param_override) {
        $run_params{$k} = $v;
    } 

    my $decode_job_id = runjobs(
        $dependant_job_ids,
        $job_name, 
        \%run_params,
        [ "mkdir -p $output_dir" ],
        train_command(
            params_template => $param_file,
            mode            => $nlplingo_mode,
        )
    );
    return [$decode_job_id];
}
1;
