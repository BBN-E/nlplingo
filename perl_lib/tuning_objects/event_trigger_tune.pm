#!/usr/bin/env perl
package event_trigger_tune;
use strict;
use warnings;


sub params_to_search {
    return (
        #"positive_weight"   => [ 1, 3, 5, 10 ],                     # [1, 3, 5]
        "batch_size"        => [ 15 ],                 # [20, 30, 50, 100]
        "num_epochs"        => [ 30 ],              # [30, 40, 50, 60, 70]
        "neighbor_distance" => [ 0 ],                            # [0, 1]
        "hidden_layers"     => [ "768"], #, "256,256", "512,512", "768,768" ], # ["256,256", "512,512", "768,768"]
        "learning_rate"     => [ 0.0001 ], #0.00001, 0.001 ]         # [0.0001, 0.00001, 0.000001]
    );
}

sub param_to_abbrev {
    return (
        "positive_weight"                  => "w",
        "batch_size"                       => "b",
        "num_epochs"                       => "e",
        "neighbor_distance"                => "n",
        "hidden_layers"                    => "hl",
        "learning_rate"                    => "lr",
        "number_of_feature_maps"           => "f",
        "cnn_filter_length"                => "fl",
        "position_embedding_vector_length" => "pemb",
        "entity_embedding_vector_length"   => "eemb"
    );
}

1;
