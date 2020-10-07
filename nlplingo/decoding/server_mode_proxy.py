# The reason this file exist is because, we need a way of dealing with multiple event trigger models problem.

raise NotImplementedError("Overall it's finished. But merge event mention logic is not clear enough")

import io
import json
from functools import partial
from uuid import uuid4

import tornado.concurrent
import tornado.httpclient
import tornado.ioloop
import tornado.web

from nlplingo.decoding.serifxml_helper import merge_event_mention_and_argument_in_serifxml

try:
    from urllib.parse import quote
except ImportError:
    # Python 2.
    from urllib import quote


async def multipart_producer(boundary, filenames, write):
    boundary_bytes = boundary.encode()

    for filename, file_buf in filenames.items():
        mtype = 'application/xml'
        buf = (
                (b'--%s\r\n' % boundary_bytes) +
                (b'Content-Disposition: form-data; name="%s"; filename="%s"\r\n' %
                 (filename.encode(), filename.encode())) +
                (b'Content-Type: %b\r\n' % mtype.encode()) +
                b'\r\n'
        )
        await write(buf)
        while True:
            # 16k at a time.
            chunk = file_buf.read(16 * 1024)
            if not chunk:
                break
            await write(chunk)

        await write(b'\r\n')

    await write(b'--%s--\r\n' % (boundary_bytes,))


async def requester(uri, serifxml_str):
    client = tornado.httpclient.AsyncHTTPClient()
    boundary = uuid4().hex
    buffer = io.BytesIO(serifxml_str.encode('utf-8'))
    buffer.seek(0)
    headers = {'Content-Type': 'multipart/form-data; boundary=%s' % boundary}
    producer = partial(multipart_producer, boundary, {'serifxml_str': buffer})
    response = await client.fetch(uri,
                                  method='POST',
                                  headers=headers,
                                  body_producer=producer)

    return response


def merge_result_json(src, dst):
    if src['status'] != 'OK' or dst['status'] != 'OK':
        return {'status': 'ERROR', 'msg': src.get('msg', '') + dst.get('msg', '')}
    dst['display_json'].extend(src['display_json'])
    dst['serifxml_str'] = merge_event_mention_and_argument_in_serifxml(src['serifxml_str'], dst['serifxml_str'])
    return dst


class MainHandler(tornado.web.RequestHandler):
    def initialize(self, endpoints):
        self.endpoints = endpoints

    async def post(self):
        fileinfo = self.request.files['serifxml_str'][0]['body']
        serifxml_str = fileinfo.decode('utf-8')
        pending_list = list()
        for endpoint in endpoints:
            f = requester(endpoint, serifxml_str)
            pending_list.append(f)
        result_list = list()
        for pending_job in pending_list:
            resp = await pending_job
            result_list.append(json.loads(resp.body.decode('utf-8')))
        target = result_list[0]
        for i in range(1, len(result_list)):
            target = merge_result_json(result_list[i], target)
        self.write(target)


def make_app(endpoints):
    return tornado.web.Application([
        (r"/event_and_argument_decode", MainHandler, {'endpoints': endpoints}),
    ])


if __name__ == "__main__":
    import sys

    endpoints = list()
    s = ""
    for arg in sys.argv:
        if "http" in arg:
            if "event_and_argument_decode" not in arg:
                arg = arg + "/event_and_argument_decode"
            endpoints.append(arg.strip())
    app = make_app(endpoints)
    app.listen(sys.argv[1])
    tornado.ioloop.IOLoop.current().start()
