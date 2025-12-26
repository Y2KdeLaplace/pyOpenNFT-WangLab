import fnmatch
import glob
import os
from pathlib import Path
import queue
import re

from loguru import logger
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
from opennft import eventrecorder as erd


class NewFileEventHandler(FileSystemEventHandler):
    def __init__(self, filepat, fq: queue.Queue, recorder: erd.EventRecorder=None):
        self.filepat = filepat
        self.fq = fq
        self.recorder = recorder

    def on_created(self, event):
        # if not event.is_directory and event.src_path.endswith(self.filepat):
        if not event.is_directory and fnmatch.fnmatch(os.path.basename(event.src_path), self.filepat):
            # if self.recorder is None:
            #     pass
            # else:
            #     # t1
            #     self.recorder.record_event(erd.Times.t1, 0, time.time())
            self.fq.put(event.src_path)


class FileWatcher():
    def __init__(self, online, watch_folder, first_filename, first_filename_txt, 
                       con, event_recorder=None):
        self.online = online
        self.watch_folder = watch_folder
        self.first_filename = first_filename
        self.first_filename_txt = first_filename_txt
        self.delimiter = con.filename_delimiter
        self.datatype = con.data_type
        self.isNatureNum = con.filename_nature_num
        self.recorder = event_recorder

        self.TR_idx = -1
        self.files_queue = queue.Queue()
        self.fs_observer = None
        self.search_string = None
    # --------------------------------------------------------------------------

    def __iter__(self):
        return self
    # --------------------------------------------------------------------------

    def __next__(self) -> Path:
        try:
            filename = self.files_queue.get(block=True, timeout=0.1)
            return Path(filename)
        except queue.Empty:
            return None
            # raise StopIteration
    # --------------------------------------------------------------------------

    def get_search_string(self, ext) -> str:

        # 1. 根据分隔符（默认为 '_'）分割文件名
        tmpl_parts = self.first_filename_txt.split(self.delimiter)
        parts = self.first_filename.split(self.delimiter)
        
        # 2.  寻找包含计数器占位符的位置 ('#')
        for i, part in enumerate(tmpl_parts):
            if '#' in part:
                self.TR_idx = i 
                break
        
        # 4. 生成 Glob 搜索字符串
        if self.TR_idx == -1:
            # 接受任意文件名
            search_string = '*' + ext
        else:
            # 如果该段包含扩展名,我们需要保留扩展名，只把前面的部分变成 *
            if parts[self.TR_idx].endswith(ext):
                parts[self.TR_idx] = '*' + ext
            else:
                # 简单粗暴地将该段替换为 * (通配符)
                # 这样就能匹配该位置的任何变化
                parts[self.TR_idx] = '*'
            
            search_string = self.delimiter.join(parts)

        return search_string
    # --------------------------------------------------------------------------

    def start_watching(self):

        path = os.path.join(self.watch_folder, self.first_filename)

        ext = re.findall(r"\.\w*$", str(path))
        if not ext:
            ext = self.datatype
        else:
            ext = ext[-1]

        self.search_string = self.get_search_string(ext)
        
        if self.online:
            path = os.path.dirname(path)
            logger.info('Online watching for {} in {}', self.search_string, path)
            event_handler = NewFileEventHandler(self.search_string, self.files_queue, self.recorder)

            self.fs_observer = Observer()
            self.fs_observer.schedule(event_handler, path, recursive=True)
            self.fs_observer.start()
        else:       
            path = os.path.join(os.path.dirname(path), self.search_string)
            logger.info(f"Offline searching for {path}")
            if self.isNatureNum:
                files = sorted(glob.glob(path), key = lambda s: int(s.split(self.delimiter)[self.TR_idx]))
            else:
                files = sorted(glob.glob(path))
        
            if not files:
                logger.info("No files found in offline mode. Check WatchFolder settings!")
                # self.stop()
                return

            self.files_queue = queue.Queue()

            for f in files:
                self.files_queue.put(f)

        # TODO move to the call
        # self.call_timer.start()
    # --------------------------------------------------------------------------
    def stop(self):

        self.fs_observer.stop()


if __name__ == '__main__':
    pass


