# Detection module: real-time face capture from surveillance camera

Based on HC-SDK console demo.

## Use two threads in this process:

* video showing thread:  Use *cv::imshow* to show real-time decoding video stream.
* face capture thread: detect, track and save face sequence to dest directory and send corresponding message to recognition module.

 
  


        
  