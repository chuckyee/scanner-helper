# Scanner Helper

Small tool that automatically identifies and saves multiple photos from a
single scanned image.


## Installation

Requires `scikit-image` and `scipy`. Install by executing:

```bash
pip install -r requirements.txt
```

The script `extract.py` is stand alone.


## Usage

Let's say you have scanned your a bunch of photos and saved them to
`IMG_1.JPEG`, `IMG_2.JPEG`, ...  Process them by executing:

```bash
./extract.py IMG_*.JPEG --outdir ./photos --errdir ./errors
```

The successfully identified photos will be written to `./photos`, cropped and
aligned. Any photo which could not be processed will be written to `./errors`.


## Notes

Threw this together to help speed up processing of hundreds of scans of family
photos for Dad. Script is not particularly robust, in particular:

 - white in photos can cause indentations in rectangular boundary detection

 - white patches in photos which cut across image will fool boundary detector

 - conversion of contour to rectangle is relies on manually-set tolerance which
   currently depends on scan resolution
