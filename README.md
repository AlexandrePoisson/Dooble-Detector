# This repo

This repo contains ressources for building an application which detects the object that appears twice in a [Dooble game](https://www.dobblegame.com).
It is possible to build an [iOS application](https://youtu.be/J3OBAjZr00k) or running python script on a Jetson Nano.
![](DoobleApp.gif)
This is mostly based on Tensorflow Object detection tutoriel.

Dooble is a trademark of Asmodee group.



# Installation

This repository contains everything required but a missing file: 

    DoobleHacker/Pods/TensorFlowLiteC/Frameworks/TensorFlowLiteC.framework/TensorFlowLiteC


This file shall be retrievied, likely using the same pod line than in the original TensorFlow tutorial.

This file was removed it because it was too big for GitHub.

    git rm --cached DoobleHacker/Pods/TensorFlowLiteC/Frameworks/TensorFlowLiteC.framework/TensorFlowLiteC
	git commit --amend -CHEAD
	git push
