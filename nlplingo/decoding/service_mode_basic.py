import sys

from nlplingo.decoding.decoder import Decoder


def create_app(params):
    import flask
    app = flask.Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
    decoder = Decoder(params)

    @app.errorhandler(Exception)
    @app.errorhandler(400)
    @app.errorhandler(404)
    @app.errorhandler(405)
    def exceptionHandler(error):
        code = getattr(error, 'code', 500)
        return flask.jsonify({"status": "ERROR"}), code

    @app.before_first_request
    def init():
        decoder.load_model()

    @app.route('/prediction_json', methods=['POST'])
    def get_prediction_json():
        try:
            serifxml_str = flask.request.files['serifxmls'].read()
            if isinstance(serifxml_str, bytes):
                serifxml_str = serifxml_str.decode('utf-8')
        except:
            import traceback
            traceback.print_exc()
            return flask.jsonify({'status': 'ERROR', 'msg': traceback.format_exc()}), 500
        try:
            list_trigger_extractor_result_collection = decoder.decode_trigger_and_argument([serifxml_str])
            prediction_json_dict = Decoder.serialize_prediction_json(list_trigger_extractor_result_collection)
            return flask.jsonify({'display_json': prediction_json_dict, 'status': 'OK'}), 200
        except:
            import traceback
            traceback.print_exc()
            return flask.jsonify({'status': 'ERROR', 'msg': traceback.format_exc()}), 500

    return app


if __name__ == "__main__":
    args = sys.argv
    params = args[1]
    port = int(args[2])
    # We override configuration here because we want to follow the original convention as much as we can
    # But generally it's not a good idea for writing configuration file on the fly.

    create_app(params).run('0.0.0.0', port)
