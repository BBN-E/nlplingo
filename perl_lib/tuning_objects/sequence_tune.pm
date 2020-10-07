#!/usr/bin/env perl
package sequence_tune;
use strict;
use warnings;


sub params_to_search {
    return (
        "batch_size"        => [ 15 ],
        "num_epochs"        => [ 30 ],
        "learning_rate"     => [ 5e-05 ],
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
