package Utils;

use Exporter;

# base class of this(Arithmetic) module
our @ISA = qw(Exporter);

# Exporting the add and subtract routine
our @EXPORT_OK = qw(cartesian);

use strict;
use warnings;

use lib ("/d4m/ears/releases/Cube2/R2019_05_24_1/install-optimize$ENV{ARCH_SUFFIX}/perl_lib");

use runjobs4;

=pod
=item cartesian()

This subroutine accepts an array of arrays and returns all possible unique
combinations as an array of arrays.

=cut
sub cartesian {
    my @C = map { [ $_ ] } @{ shift @_ };

    foreach (@_) {
        my @A = @$_;

        @C = map { my $n = $_; map { [ $n, @$_ ] } @C } @A;
    }

    return @C;
}

sub gather_docs_to_list {
    my (%args) = @_;

    my $dependant_job_ids = $args{dependant_job_ids};
    my $instance_name = $args{instance_name};
    my $gather_folder = $args{gather_folder};
    my $list_file = $args{list_file};
    my $job_name = $args{job_name};
    my $search_string = $args{search_string};
    my $LINUX_CPU_QUEUE = $args{linux_cpu_queue};

    my $gather_job_id = runjobs(
        $dependant_job_ids,
        "cpu/$job_name/utils/gather_docs_to_list/$instance_name",
        {
            BATCH_QUEUE => $LINUX_CPU_QUEUE
        },
        [
            "find $gather_folder -type f -name \"" . $search_string . "\" -exec readlink -f {} \\; " .
                "| sort -u > $list_file"
        ]
    );
    return [$gather_job_id];
}

sub make_output_dir {
    my $dir = $_[0];
    my $job_name = $_[1];
    my $job_depdendencies = $_[2];
    my $job_id = runjobs4::runjobs(
        $job_depdendencies, $job_name,
        {
            SCRIPT => 1
        },
        [ "mkdir -p $dir" ]
    );
    return $dir,[$job_id];
}

sub split_file_list_with_num_of_batches {
    my %args = @_;
    my $PYTHON = $args{PYTHON};
    my $CREATE_FILELIST_PY_PATH = $args{CREATE_FILELIST_PY_PATH};
    my $dependant_job_ids = $args{dependant_job_ids};
    my $num_of_batches = $args{num_of_batches};
    my $job_prefix = $args{job_prefix};
    my $list_file_path = $args{list_file_path};
    my $output_file_prefix = $args{output_file_prefix};
    my $suffix = $args{suffix};
    my $create_filelist_jobid = runjobs4::runjobs(
        $dependant_job_ids, $job_prefix . "split_filelist_with_num_of_batches_" . $num_of_batches,
        {
            SCRIPT => 1
        },
        [ "$PYTHON $CREATE_FILELIST_PY_PATH --list_file_path \"$list_file_path\" --output_list_prefix \"$output_file_prefix\" --num_of_batches $num_of_batches --suffix \"$suffix\"" ]
    );
    my @file_list_at_disk = ();
    for (my $i = 0; $i < $num_of_batches; ++$i) {
        push(@file_list_at_disk, $output_file_prefix . $i . $suffix);
    }
    return([ $create_filelist_jobid ], @file_list_at_disk);
}

sub split_file_for_processing {
    my $split_jobname = $_[0];
    my $bf = $_[1];
    my $bp = $_[2];
    my $bs = $_[3];

    open my $fh, "<", $bf or die "could not open $bf: $!";
    my $num_files = 0;
    $num_files++ while <$fh>;
    my $njobs = int($num_files / $bs) + 1;
    if ($num_files % $bs == 0) {
        $njobs--;
    }
    print "File $bf will be broken into $njobs batches of approximate size $bs\n";
    my $jobid = runjobs4::runjobs([], "$split_jobname",
        {
            SCRIPT => 1,
        },
        "/usr/bin/split -d -a 5 -l $bs $bf $bp");

    return($njobs, $jobid);
}

1;
