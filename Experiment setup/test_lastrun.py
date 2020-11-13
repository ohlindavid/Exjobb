#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2020.2.5),
    on November 13, 2020, at 10:03
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

from __future__ import absolute_import, division

from psychopy import locale_setup
from psychopy import prefs
from psychopy import sound, gui, visual, core, data, event, logging, clock
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle
import os  # handy system and path functions
import sys  # to get file system encoding

from psychopy.hardware import keyboard



# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)

# Store info about the experiment session
psychopyVersion = '2020.2.5'
expName = 'DecodingEmotions'  # from the Builder filename that created this script
expInfo = {'participant': '', 'session': '001'}
dlg = gui.DlgFromDict(dictionary=expInfo, sort_keys=False, title=expName)
if dlg.OK == False:
    core.quit()  # user pressed cancel
expInfo['date'] = data.getDateStr()  # add a simple timestamp
expInfo['expName'] = expName
expInfo['psychopyVersion'] = psychopyVersion

# Data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
filename = _thisDir + os.sep + u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])

# An ExperimentHandler isn't essential but helps with data saving
thisExp = data.ExperimentHandler(name=expName, version='',
    extraInfo=expInfo, runtimeInfo=None,
    originPath='C:\\Users\\Oskar\\Documents\\GitHub\\exjobb\\Experiment setup\\test_lastrun.py',
    savePickle=True, saveWideText=True,
    dataFileName=filename)
# save a log file for detail verbose info
logFile = logging.LogFile(filename+'.log', level=logging.EXP)
logging.console.setLevel(logging.WARNING)  # this outputs to the screen, not a file

endExpNow = False  # flag for 'escape' or other condition => quit the exp
frameTolerance = 0.001  # how close to onset before 'same' frame

# Start Code - component code to be run before the window creation

# Setup the Window
win = visual.Window(
    size=[1920, 1080], fullscr=True, screen=0, 
    winType='pyglet', allowGUI=True, allowStencil=False,
    monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
    blendMode='avg', useFBO=True, 
    units='height')
# store frame rate of monitor if we can measure it
expInfo['frameRate'] = win.getActualFrameRate()
if expInfo['frameRate'] != None:
    frameDur = 1.0 / round(expInfo['frameRate'])
else:
    frameDur = 1.0 / 60.0  # could not measure, so guess

# create a default keyboard (e.g. to check for escape)
defaultKeyboard = keyboard.Keyboard()

# Initialize components for Routine "Training"
TrainingClock = core.Clock()
text = visual.TextStim(win=win, name='text',
    text='default text',
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='red', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
image = visual.ImageStim(
    win=win,
    name='image', 
    image='sin', mask=None,
    ori=0, pos=(0, 0), size=(0.5, 0.5),
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-1.0)
image2 = visual.ImageStim(
    win=win,
    name='image2', 
    image='sin', mask=None,
    ori=0, pos=(0, 0), size=(0.5, 0.5),
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-2.0)
text2 = visual.TextStim(win=win, name='text2',
    text='default text',
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='red', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-3.0);

# Initialize components for Routine "Distraction"
DistractionClock = core.Clock()
Distraction_text = visual.TextStim(win=win, name='Distraction_text',
    text='Try to solve the arthimetic tasks on the paper in front of you, until the screen shows the next instructions.',
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='red', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);

# Initialize components for Routine "Instructions_for_retrival"
Instructions_for_retrivalClock = core.Clock()
Instructions = visual.TextStim(win=win, name='Instructions',
    text='Now you will be presented with a word. \n\nTry to image the image shown in conjugtion to it. \n\nAfter a small time for each word you will be able to choose if the word was positive negative or neutral and if it was a face or a scene.',
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='red', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);

# Initialize components for Routine "Retrival"
RetrivalClock = core.Clock()
Word_cue = visual.TextStim(win=win, name='Word_cue',
    text=word,
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='red', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
memory_check = visual.RatingScale(win=win, name='memory_check', marker='triangle', size=1.0, pos=[0.0, -0.4], choices=['Positive face', 'Positive scene', 'Negative face', 'Negative scene', 'Neutral face', 'Neutral scene', "I don't remember"], tickHeight=-1)

# Create some handy timers
globalClock = core.Clock()  # to track the time since experiment started
routineTimer = core.CountdownTimer()  # to track time remaining of each (non-slip) routine 

# set up handler to look after randomisation of conditions etc
trials_3 = data.TrialHandler(nReps=number_of_runs, method='sequential', 
    extraInfo=expInfo, originPath=-1,
    trialList=[None],
    seed=None, name='trials_3')
thisExp.addLoop(trials_3)  # add the loop to the experiment
thisTrial_3 = trials_3.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisTrial_3.rgb)
if thisTrial_3 != None:
    for paramName in thisTrial_3:
        exec('{} = thisTrial_3[paramName]'.format(paramName))

for thisTrial_3 in trials_3:
    currentLoop = trials_3
    # abbreviate parameter names if possible (e.g. rgb = thisTrial_3.rgb)
    if thisTrial_3 != None:
        for paramName in thisTrial_3:
            exec('{} = thisTrial_3[paramName]'.format(paramName))
    
    # set up handler to look after randomisation of conditions etc
    trials = data.TrialHandler(nReps=learning_batch_size, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('conditionsTest.csv'),
        seed=None, name='trials')
    thisExp.addLoop(trials)  # add the loop to the experiment
    thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial:
            exec('{} = thisTrial[paramName]'.format(paramName))
    
    for thisTrial in trials:
        currentLoop = trials
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                exec('{} = thisTrial[paramName]'.format(paramName))
        
        # ------Prepare to start Routine "Training"-------
        continueRoutine = True
        routineTimer.add(10.500000)
        # update component parameters for each repeat
        text.setText(word)
        image.setImage(image)
        image2.setImage(image)
        text2.setText(word)
        # keep track of which components have finished
        TrainingComponents = [text, image, image2, text2]
        for thisComponent in TrainingComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        TrainingClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
        frameN = -1
        
        # -------Run Routine "Training"-------
        while continueRoutine and routineTimer.getTime() > 0:
            # get current time
            t = TrainingClock.getTime()
            tThisFlip = win.getFutureFlipTime(clock=TrainingClock)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text* updates
            if text.status == NOT_STARTED and tThisFlip >= 1.0-frameTolerance:
                # keep track of start time/frame for later
                text.frameNStart = frameN  # exact frame index
                text.tStart = t  # local t and not account for scr refresh
                text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
                text.setAutoDraw(True)
            if text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text.tStartRefresh + 2.5-frameTolerance:
                    # keep track of stop time/frame for later
                    text.tStop = t  # not accounting for scr refresh
                    text.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(text, 'tStopRefresh')  # time at next scr refresh
                    text.setAutoDraw(False)
            
            # *image* updates
            if image.status == NOT_STARTED and tThisFlip >= 4.5-frameTolerance:
                # keep track of start time/frame for later
                image.frameNStart = frameN  # exact frame index
                image.tStart = t  # local t and not account for scr refresh
                image.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(image, 'tStartRefresh')  # time at next scr refresh
                image.setAutoDraw(True)
            if image.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > image.tStartRefresh + 2.5-frameTolerance:
                    # keep track of stop time/frame for later
                    image.tStop = t  # not accounting for scr refresh
                    image.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(image, 'tStopRefresh')  # time at next scr refresh
                    image.setAutoDraw(False)
            
            # *image2* updates
            if image2.status == NOT_STARTED and tThisFlip >= 8-frameTolerance:
                # keep track of start time/frame for later
                image2.frameNStart = frameN  # exact frame index
                image2.tStart = t  # local t and not account for scr refresh
                image2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(image2, 'tStartRefresh')  # time at next scr refresh
                image2.setAutoDraw(True)
            if image2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > image2.tStartRefresh + 2.5-frameTolerance:
                    # keep track of stop time/frame for later
                    image2.tStop = t  # not accounting for scr refresh
                    image2.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(image2, 'tStopRefresh')  # time at next scr refresh
                    image2.setAutoDraw(False)
            
            # *text2* updates
            if text2.status == NOT_STARTED and tThisFlip >= 8-frameTolerance:
                # keep track of start time/frame for later
                text2.frameNStart = frameN  # exact frame index
                text2.tStart = t  # local t and not account for scr refresh
                text2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text2, 'tStartRefresh')  # time at next scr refresh
                text2.setAutoDraw(True)
            if text2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text2.tStartRefresh + 2.5-frameTolerance:
                    # keep track of stop time/frame for later
                    text2.tStop = t  # not accounting for scr refresh
                    text2.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(text2, 'tStopRefresh')  # time at next scr refresh
                    text2.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in TrainingComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # -------Ending Routine "Training"-------
        for thisComponent in TrainingComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        trials.addData('text.started', text.tStartRefresh)
        trials.addData('text.stopped', text.tStopRefresh)
        trials.addData('image.started', image.tStartRefresh)
        trials.addData('image.stopped', image.tStopRefresh)
        trials.addData('image2.started', image2.tStartRefresh)
        trials.addData('image2.stopped', image2.tStopRefresh)
        trials.addData('text2.started', text2.tStartRefresh)
        trials.addData('text2.stopped', text2.tStopRefresh)
    # completed learning_batch_size repeats of 'trials'
    
    
    # ------Prepare to start Routine "Distraction"-------
    continueRoutine = True
    # update component parameters for each repeat
    # keep track of which components have finished
    DistractionComponents = [Distraction_text]
    for thisComponent in DistractionComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    DistractionClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "Distraction"-------
    while continueRoutine:
        # get current time
        t = DistractionClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=DistractionClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *Distraction_text* updates
        if Distraction_text.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
            # keep track of start time/frame for later
            Distraction_text.frameNStart = frameN  # exact frame index
            Distraction_text.tStart = t  # local t and not account for scr refresh
            Distraction_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Distraction_text, 'tStartRefresh')  # time at next scr refresh
            Distraction_text.setAutoDraw(True)
        if Distraction_text.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > Distraction_text.tStartRefresh + distraction_time-frameTolerance:
                # keep track of stop time/frame for later
                Distraction_text.tStop = t  # not accounting for scr refresh
                Distraction_text.frameNStop = frameN  # exact frame index
                win.timeOnFlip(Distraction_text, 'tStopRefresh')  # time at next scr refresh
                Distraction_text.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in DistractionComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "Distraction"-------
    for thisComponent in DistractionComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    trials_3.addData('Distraction_text.started', Distraction_text.tStartRefresh)
    trials_3.addData('Distraction_text.stopped', Distraction_text.tStopRefresh)
    # the Routine "Distraction" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # ------Prepare to start Routine "Instructions_for_retrival"-------
    continueRoutine = True
    # update component parameters for each repeat
    # keep track of which components have finished
    Instructions_for_retrivalComponents = [Instructions]
    for thisComponent in Instructions_for_retrivalComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    Instructions_for_retrivalClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "Instructions_for_retrival"-------
    while continueRoutine:
        # get current time
        t = Instructions_for_retrivalClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=Instructions_for_retrivalClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *Instructions* updates
        if Instructions.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
            # keep track of start time/frame for later
            Instructions.frameNStart = frameN  # exact frame index
            Instructions.tStart = t  # local t and not account for scr refresh
            Instructions.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Instructions, 'tStartRefresh')  # time at next scr refresh
            Instructions.setAutoDraw(True)
        if Instructions.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > Instructions.tStartRefresh + instruction_time-frameTolerance:
                # keep track of stop time/frame for later
                Instructions.tStop = t  # not accounting for scr refresh
                Instructions.frameNStop = frameN  # exact frame index
                win.timeOnFlip(Instructions, 'tStopRefresh')  # time at next scr refresh
                Instructions.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Instructions_for_retrivalComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "Instructions_for_retrival"-------
    for thisComponent in Instructions_for_retrivalComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    trials_3.addData('Instructions.started', Instructions.tStartRefresh)
    trials_3.addData('Instructions.stopped', Instructions.tStopRefresh)
    # the Routine "Instructions_for_retrival" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trials_2 = data.TrialHandler(nReps=batch_learn_size, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('conditionsTest.csv'),
        seed=None, name='trials_2')
    thisExp.addLoop(trials_2)  # add the loop to the experiment
    thisTrial_2 = trials_2.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial_2.rgb)
    if thisTrial_2 != None:
        for paramName in thisTrial_2:
            exec('{} = thisTrial_2[paramName]'.format(paramName))
    
    for thisTrial_2 in trials_2:
        currentLoop = trials_2
        # abbreviate parameter names if possible (e.g. rgb = thisTrial_2.rgb)
        if thisTrial_2 != None:
            for paramName in thisTrial_2:
                exec('{} = thisTrial_2[paramName]'.format(paramName))
        
        # ------Prepare to start Routine "Retrival"-------
        continueRoutine = True
        # update component parameters for each repeat
        memory_check.reset()
        # keep track of which components have finished
        RetrivalComponents = [Word_cue, memory_check]
        for thisComponent in RetrivalComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        RetrivalClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
        frameN = -1
        
        # -------Run Routine "Retrival"-------
        while continueRoutine:
            # get current time
            t = RetrivalClock.getTime()
            tThisFlip = win.getFutureFlipTime(clock=RetrivalClock)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *Word_cue* updates
            if Word_cue.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                # keep track of start time/frame for later
                Word_cue.frameNStart = frameN  # exact frame index
                Word_cue.tStart = t  # local t and not account for scr refresh
                Word_cue.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Word_cue, 'tStartRefresh')  # time at next scr refresh
                Word_cue.setAutoDraw(True)
            if Word_cue.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > Word_cue.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    Word_cue.tStop = t  # not accounting for scr refresh
                    Word_cue.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(Word_cue, 'tStopRefresh')  # time at next scr refresh
                    Word_cue.setAutoDraw(False)
            # *memory_check* updates
            if memory_check.status == NOT_STARTED and t >= 5-frameTolerance:
                # keep track of start time/frame for later
                memory_check.frameNStart = frameN  # exact frame index
                memory_check.tStart = t  # local t and not account for scr refresh
                memory_check.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(memory_check, 'tStartRefresh')  # time at next scr refresh
                memory_check.setAutoDraw(True)
            continueRoutine &= memory_check.noResponse  # a response ends the trial
            
            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in RetrivalComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # -------Ending Routine "Retrival"-------
        for thisComponent in RetrivalComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        trials_2.addData('Word_cue.started', Word_cue.tStartRefresh)
        trials_2.addData('Word_cue.stopped', Word_cue.tStopRefresh)
        # store data for trials_2 (TrialHandler)
        trials_2.addData('memory_check.response', memory_check.getRating())
        trials_2.addData('memory_check.rt', memory_check.getRT())
        trials_2.addData('memory_check.started', memory_check.tStart)
        trials_2.addData('memory_check.stopped', memory_check.tStop)
        # the Routine "Retrival" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
    # completed batch_learn_size repeats of 'trials_2'
    
    thisExp.nextEntry()
    
# completed number_of_runs repeats of 'trials_3'


# Flip one final time so any remaining win.callOnFlip() 
# and win.timeOnFlip() tasks get executed before quitting
win.flip()

# these shouldn't be strictly necessary (should auto-save)
thisExp.saveAsWideText(filename+'.csv', delim='auto')
thisExp.saveAsPickle(filename)
logging.flush()
# make sure everything is closed down
thisExp.abort()  # or data files will save again on exit
win.close()
core.quit()
