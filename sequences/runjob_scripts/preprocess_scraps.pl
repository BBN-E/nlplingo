
            (my $ministage_batch_dir, my $mkdir_batch_jobs) = Utils::make_output_dir("$processing_dir/$stage_name/$mini_stage_name/$task/batch_file", "$JOB_NAME/$stage_name/$mini_stage_name/$task/mkdir_stage_batch_processing", \@empty);

            my $pyserif_dep_jobid = runjobs4::runjobs(
            $mkdir_batch_jobs, "$JOB_NAME/$stage_name/$mini_stage_name/" . $task . "/pyserif_$batch",
            $pyserif_template,
            [ "mkdir -p $batch_output_folder" ],
            [ "env PYTHONPATH=$nlplingo_root:$textopen_root/src/python " .
                "KERAS_BACKEND=tensorflow " .
                "$ANACONDA_ROOT/envs/$CONDA_ENV_NAME_FOR_BERT_CPU/bin/python $nlplingo_root/nlplingo/tasks/driver.py --params $train_json --mode preprocess --task $task"]
#            [ "env PYTHONPATH=$nlplingo_root:$textopen_root/src/python " .
#                "KERAS_BACKEND=tensorflow " .
#                "$ANACONDA_ROOT/envs/$CONDA_ENV_NAME_FOR_BERT_CPU/bin/python $nlplingo_root/nlplingo/tasks/train_test.py --params", $task . "_preprocess.json", "--mode preprocess --task $task"]
             );

            my @split_jobs = ();

            (my $create_filelist_train_jobid_$task, undef
            ) = Utils::split_file_list_with_num_of_batches(
                PYTHON                  => $PYTHON3,
                CREATE_FILELIST_PY_PATH => $CREATE_FILELIST_PY_PATH,
                dependant_job_ids       => $mkdir_batch_jobs,
                job_prefix              => "$JOB_NAME/$stage_name/$mini_stage_name/$task",
                num_of_batches          => $NUM_OF_BATCHES_GLOBAL,
                list_file_path          => $train_file,
                output_file_prefix      => $ministage_batch_dir . "/batch_train",
                suffix                  => ".list",
            );


            for (my $batch = 0; $batch < $NUM_OF_BATCHES_GLOBAL; $batch++) {
                my $batch_file = "$ministage_batch_dir/batch_$batch.list";
                my $batch_output_folder = "$ministage_processing_dir/$batch/output";
                my $pyserif_dep_jobid = runjobs4::runjobs(
                $mkdir_batch_jobs, "$JOB_NAME/$stage_name/$mini_stage_name/" . $task . "/pyserif_$batch",
                $pyserif_template,
                [ "mkdir -p $batch_output_folder" ],
                [ "env PYTHONPATH=$nlplingo_root:$textopen_root/src/python " .
                    "KERAS_BACKEND=tensorflow " .
                    "$ANACONDA_ROOT/envs/$CONDA_ENV_NAME_FOR_BERT_CPU/bin/python $nlplingo_root/nlplingo/tasks/train_test.py --params", $task . "_preprocess.json", "--mode preprocess --task $task"]
                 );
                push(@split_jobs, $pyserif_dep_jobid);
            }
        }


            (my $ministage_batch_dir, my $mkdir_batch_jobs) = Utils::make_output_dir("$processing_dir/$stage_name/$mini_stage_name/$task/batch_file", "$JOB_NAME/$stage_name/$mini_stage_name/$task/mkdir_stage_batch_processing", \@empty);




    {
        my $mini_stage_name = "serialize";
        foreach my $task (@tasks) {
            (my $ministage_processing_dir, undef) = Utils::make_output_dir("$processing_dir/$stage_name/$mini_stage_name/$task", "$JOB_NAME/$stage_name/$mini_stage_name/mkdir_stage_processing/$task", []);
            (my $ministage_batch_dir, my $mkdir_batch_jobs) = Utils::make_output_dir("$processing_dir/$stage_name/$mini_stage_name/$task/batch_file", "$JOB_NAME/$stage_name/$mini_stage_name/$task/mkdir_stage_batch_processing", \@dep_gen_task_list_ids);

            my @split_jobs = ();

            (my $create_filelist_jobid, undef
            ) = Utils::split_file_list_with_num_of_batches(
                PYTHON                  => $PYTHON3,
                CREATE_FILELIST_PY_PATH => $CREATE_FILELIST_PY_PATH,
                dependant_job_ids       => $mkdir_batch_jobs,
                job_prefix              => "$JOB_NAME/$stage_name/$mini_stage_name/$task",
                num_of_batches          => $NUM_OF_BATCHES_GLOBAL,
                list_file_path          => $aggregate_serif_output_dir/$task . ".list",
                output_file_prefix      => $ministage_batch_dir . "/batch_" . $task. "_",
                suffix                  => ".list",
            );
        }
    }


            for (my $batch = 0; $batch < $NUM_OF_BATCHES_GLOBAL; $batch++) {
                my $batch_file = "$ministage_batch_dir/batch_$task_$batch.list";
                my $batch_output_folder = "$ministage_processing_dir/$batch/$task/output";
                my $pyserif_dep_jobid = runjobs4::runjobs(
                $mkdir_batch_jobs, "$JOB_NAME/$stage_name/$mini_stage_name/" . $task . "/pyserif_$batch",
                $pyserif_template,
                [ "mkdir -p $batch_output_folder" ],
                [ "env PYTHONPATH=$nlplingo_root:$textopen_root/src/python " .
                    "KERAS_BACKEND=tensorflow " .
                    "$ANACONDA_ROOT/envs/$CONDA_ENV_NAME_FOR_BERT_CPU/bin/python  $nlplingo_root/nlplingo/modes/preprocess/serialize.py --task $task --filelist $batch_file --output_directory $batch_output_folder --batch_num $batch"]
                );
                push(@split_jobs, $pyserif_dep_jobid);
            }
