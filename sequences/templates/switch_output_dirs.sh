
input_params=+params+
out_dir=+output_dir+
search_str="+search_str+"

cp $input_params $out_dir/train.json

sed -i "s~\"save_model\": false~\"save_model\": true~g" $out_dir/train.json
sed -i "s~\": \".*\?$search_str.*/\([^/]\+\)\"~\": \"$out_dir/\1\"~g" $out_dir/train.json

