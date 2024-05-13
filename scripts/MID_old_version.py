#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2023.2.3),
    on Tue Apr 30 15:16:44 2024
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

from pylsl import StreamInfo, StreamOutlet
# Set up LabStreamingLayer stream.
info = StreamInfo(name='PsychoPy_LSL_MID', type='Markers', channel_count=1, nominal_srate=0, channel_format='string', source_id='psy_marker')
outlet = StreamOutlet(info)  # Broadcast the stream.

# Run 'Before Experiment' code from RT_list_code
rt_list = []
# --- Setup global variables (available in all functions) ---
# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# Store info about the experiment session
psychopyVersion = '2023.2.3'
expName = 'MID_python'  # from the Builder filename that created this script
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date': data.getDateStr(),  # add a simple timestamp
    'expName': expName,
    'psychopyVersion': psychopyVersion,
}


def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # temporarily remove keys which the dialog doesn't need to show
    poppedKeys = {
        'date': expInfo.pop('date', data.getDateStr()),
        'expName': expInfo.pop('expName', expName),
        'psychopyVersion': expInfo.pop('psychopyVersion', psychopyVersion),
    }
    # show participant info dialog
    dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # restore hidden keys
    expInfo.update(poppedKeys)
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='/Users/charlotte/Dropbox/Charite_PhD/tasks/MID_final/MID2.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # this outputs to the screen, not a file
    logging.console.setLevel(logging.EXP)
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log', level=logging.EXP)
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=[1920, 1080], fullscr=True, screen=1,
            winType='pyglet', allowStencil=False,
            monitor='testMonitor', color=[-1,-1,-1], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height'
        )
        if expInfo is not None:
            # store frame rate of monitor if we can measure it
            expInfo['frameRate'] = win.getActualFrameRate()
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [-1,-1,-1]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    win.mouseVisible = False
    win.hideMessage()
    return win


def setupInputs(expInfo, thisExp, win):
    """
    Setup whatever inputs are available (mouse, keyboard, eyetracker, etc.)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    dict
        Dictionary of input devices by name.
    """
    # --- Setup input devices ---
    inputs = {}
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    ioSession = '1'
    if 'session' in expInfo:
        ioSession = str(expInfo['session'])
    ioServer = io.launchHubServer(window=win, **ioConfig)
    eyetracker = None
    
    # create a default keyboard (e.g. to check for escape)
    defaultKeyboard = keyboard.Keyboard(backend='iohub')
    # return inputs dict
    return {
        'ioServer': ioServer,
        'defaultKeyboard': defaultKeyboard,
        'eyetracker': eyetracker,
    }

def pauseExperiment(thisExp, inputs=None, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    inputs : dict
        Dictionary of input devices by name.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # prevent components from auto-drawing
    win.stashAutoDraw()
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # make sure we have a keyboard
        if inputs is None:
            inputs = {
                'defaultKeyboard': keyboard.Keyboard(backend='ioHub')
            }
        # check for quit (typically the Esc key)
        if inputs['defaultKeyboard'].getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win, inputs=inputs)
        # flip the screen
        win.flip()
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, inputs=inputs, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # restore auto-drawn components
    win.retrieveAutoDraw()
    # reset any timers
    for timer in timers:
        timer.reset()


def run(expInfo, thisExp, win, inputs, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    inputs : dict
        Dictionary of input devices by name.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = inputs['ioServer']
    defaultKeyboard = inputs['defaultKeyboard']
    eyetracker = inputs['eyetracker']
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "RT_measure_start" ---
    RT_measure_start_text = visual.TextStim(win=win, name='RT_measure_start_text',
        text='Initiale Reaktionsgeschwindigkeit. \n\nWenn das grüne Kreuz rechts oder links erscheint, drücken Sie so schnell wie möglich die rechte oder linke Taste. \n \nDrücken Sie zum Starten die rechte Taste.\n',
        font='Open Sans',
        pos=[0,0], height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    RT_startscreen_press = keyboard.Keyboard()
    
    # --- Initialize components for Routine "RT_measure_fixation" ---
    rt_measure_fixation = visual.ShapeStim(
        win=win, name='rt_measure_fixation', vertices='cross',
        size=(0.01, 0.01),
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "RT_measure_target" ---
    RT_target_fig = visual.ShapeStim(
        win=win, name='RT_target_fig', vertices='cross',
        size=(0.25, 0.25),
        ori=0.0, pos=[0,0], anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='green', fillColor='green',
        opacity=None, depth=0.0, interpolate=True)
    RT_measure_response = keyboard.Keyboard()
    
    # --- Initialize components for Routine "RT_measure_end" ---
    RT_measure_endtext = visual.TextStim(win=win, name='RT_measure_endtext',
        text='Ende des Tests zur Initiale Reaktionsgeschwindigkeit.',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "StartScreen_training" ---
    StartScreen_ButtonPress_training = keyboard.Keyboard()
    StartScreen_text_training = visual.TextStim(win=win, name='StartScreen_text_training',
        text='Jetzt folgen 12 Übungssitzungen.\n\nDrücken Sie zum Starten die rechte Taste.',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "initial_iti_training" ---
    initial_iti_fig_training = visual.ShapeStim(
        win=win, name='initial_iti_fig_training', vertices='cross',
        size=(0.01, 0.01),
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    # Run 'Begin Experiment' code from initial_iti_code_training
    # Variables used for calculating target presentation time 
    # and score count in practice loop
    reward_counter_training = 0;
    #target_pres_time = 0.8;
    trial_num = 0;
    calibration_accuracy = [];
    
    # --- Initialize components for Routine "CuePresentation_training" ---
    CueCircle_training = visual.ShapeStim(
        win=win, name='CueCircle_training',
        size=(0.25, 0.25), vertices='circle',
        ori=0.0, pos=[0,0], anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    EarlyPressCue_training = keyboard.Keyboard()
    
    # --- Initialize components for Routine "Fixation2000_training" ---
    Fixation2000_fig_training = visual.ShapeStim(
        win=win, name='Fixation2000_fig_training', vertices='cross',
        size=(0.01, 0.01),
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    EarlyPressFixation_training = keyboard.Keyboard()
    
    # --- Initialize components for Routine "TargetPresentation_training" ---
    TargetPresentation_fig_training = visual.ShapeStim(
        win=win, name='TargetPresentation_fig_training',
        size=(0.25, 0.25), vertices='circle',
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-1.0, interpolate=True)
    ButtonPressTarget_training = keyboard.Keyboard()
    
    # --- Initialize components for Routine "FeedbackCode_training" ---
    text_Feedback_training = visual.TextStim(win=win, name='text_Feedback_training',
        text='',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    text_treatCounter_training = visual.TextStim(win=win, name='text_treatCounter_training',
        text='',
        font='Open Sans',
        pos=(0, -0.25), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "ITI500_training" ---
    trial_ITI_training = visual.ShapeStim(
        win=win, name='trial_ITI_training', vertices='cross',
        size=(0.01, 0.01),
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "EndScreen_training" ---
    EndScreenText_training = visual.TextStim(win=win, name='EndScreenText_training',
        text='Herzlichen Glückwunsch, Sie haben die Trainingsphase erfolgreich beendet!\n \nBitte geben Sie der Versuchsleiterin Bescheid, um fortzufahren.',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    endTraining_startMain = keyboard.Keyboard()
    
    # --- Initialize components for Routine "WelcomeScreen" ---
    StartScreen_ButtonPress = keyboard.Keyboard()
    textStartScreen = visual.TextStim(win=win, name='textStartScreen',
        text='Beginn der Aufgabe. \n\nDrücken Sie die rechte Taste, um mit der Aufgabe zu beginnen.\n\nViel Erfolg!',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "initial_iti" ---
    first_ITI_fig = visual.ShapeStim(
        win=win, name='first_ITI_fig', vertices='cross',
        size=(0.01, 0.01),
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    # Run 'Begin Experiment' code from leadin_ITI_code
    # Variables used for calculating target presentation time 
    # and score count
    treat_counter = 0;
    #target_pres_time = 0.4; !!Set in practice loop
    #trial_num = 0;
    #calibration_accuracy = []; !!Set in practice loop
    
    # --- Initialize components for Routine "CuePresentation" ---
    CueCircle = visual.ShapeStim(
        win=win, name='CueCircle',
        size=(0.25, 0.25), vertices='circle',
        ori=0.0, pos=[0,0], anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    EarlyPressCue = keyboard.Keyboard()
    
    # --- Initialize components for Routine "Fixation2000" ---
    FixationScreen = visual.ShapeStim(
        win=win, name='FixationScreen', vertices='cross',
        size=(0.01, 0.01),
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    EarlyPressFixation = keyboard.Keyboard()
    
    # --- Initialize components for Routine "TargetPresentation" ---
    TargetPresentationScreen = visual.ShapeStim(
        win=win, name='TargetPresentationScreen',
        size=(0.25, 0.25), vertices='circle',
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-1.0, interpolate=True)
    ButtonPressTarget = keyboard.Keyboard()
    
    # --- Initialize components for Routine "FeedbackCode" ---
    text_Feedback = visual.TextStim(win=win, name='text_Feedback',
        text='',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    text_treatCounter = visual.TextStim(win=win, name='text_treatCounter',
        text='',
        font='Open Sans',
        pos=(0, -0.25), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "ITI500" ---
    trial_ITI = visual.ShapeStim(
        win=win, name='trial_ITI', vertices='cross',
        size=(0.01, 0.01),
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "EndScreen" ---
    EndScreenText = visual.TextStim(win=win, name='EndScreenText',
        text='Ende der Aufgabe.\n\nDanke für die Teilnahme.',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # create some handy timers
    if globalClock is None:
        globalClock = core.Clock()  # to track the time since experiment started
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    routineTimer = core.Clock()  # to track time remaining of each (possibly non-slip) routine
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6)
    
    # --- Prepare to start Routine "RT_measure_start" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('RT_measure_start.started', globalClock.getTime())
    RT_measure_start_text.setPos((0, 0))
    RT_startscreen_press.keys = []
    RT_startscreen_press.rt = []
    _RT_startscreen_press_allKeys = []
    # keep track of which components have finished
    RT_measure_startComponents = [RT_measure_start_text, RT_startscreen_press]
    for thisComponent in RT_measure_startComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "RT_measure_start" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *RT_measure_start_text* updates
        
        # if RT_measure_start_text is starting this frame...
        if RT_measure_start_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            RT_measure_start_text.frameNStart = frameN  # exact frame index
            RT_measure_start_text.tStart = t  # local t and not account for scr refresh
            RT_measure_start_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(RT_measure_start_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'RT_measure_start_text.started')
            # update status
            RT_measure_start_text.status = STARTED
            RT_measure_start_text.setAutoDraw(True)
        
        # if RT_measure_start_text is active this frame...
        if RT_measure_start_text.status == STARTED:
            # update params
            pass
        
        # *RT_startscreen_press* updates
        waitOnFlip = False
        
        # if RT_startscreen_press is starting this frame...
        if RT_startscreen_press.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            RT_startscreen_press.frameNStart = frameN  # exact frame index
            RT_startscreen_press.tStart = t  # local t and not account for scr refresh
            RT_startscreen_press.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(RT_startscreen_press, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'RT_startscreen_press.started')
            # update status
            RT_startscreen_press.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(RT_startscreen_press.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(RT_startscreen_press.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if RT_startscreen_press.status == STARTED and not waitOnFlip:
            theseKeys = RT_startscreen_press.getKeys(keyList=['right'], ignoreKeys=["escape"], waitRelease=False)
            _RT_startscreen_press_allKeys.extend(theseKeys)
            if len(_RT_startscreen_press_allKeys):
                RT_startscreen_press.keys = _RT_startscreen_press_allKeys[0].name  # just the first key pressed
                RT_startscreen_press.rt = _RT_startscreen_press_allKeys[0].rt
                RT_startscreen_press.duration = _RT_startscreen_press_allKeys[0].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in RT_measure_startComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "RT_measure_start" ---
    for thisComponent in RT_measure_startComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('RT_measure_start.stopped', globalClock.getTime())
    # check responses
    if RT_startscreen_press.keys in ['', [], None]:  # No response was made
        RT_startscreen_press.keys = None
    thisExp.addData('RT_startscreen_press.keys',RT_startscreen_press.keys)
    if RT_startscreen_press.keys != None:  # we had a response
        thisExp.addData('RT_startscreen_press.rt', RT_startscreen_press.rt)
        thisExp.addData('RT_startscreen_press.duration', RT_startscreen_press.duration)
    thisExp.nextEntry()
    # the Routine "RT_measure_start" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    RT_measure_loop = data.TrialHandler(nReps=5.0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('MID_RT_conditions.xlsx'),
        seed=None, name='RT_measure_loop')
    thisExp.addLoop(RT_measure_loop)  # add the loop to the experiment
    thisRT_measure_loop = RT_measure_loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisRT_measure_loop.rgb)
    if thisRT_measure_loop != None:
        for paramName in thisRT_measure_loop:
            globals()[paramName] = thisRT_measure_loop[paramName]
    
    for thisRT_measure_loop in RT_measure_loop:
        currentLoop = RT_measure_loop
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisRT_measure_loop.rgb)
        if thisRT_measure_loop != None:
            for paramName in thisRT_measure_loop:
                globals()[paramName] = thisRT_measure_loop[paramName]
        
        # --- Prepare to start Routine "RT_measure_fixation" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('RT_measure_fixation.started', globalClock.getTime())
        # keep track of which components have finished
        RT_measure_fixationComponents = [rt_measure_fixation]
        for thisComponent in RT_measure_fixationComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "RT_measure_fixation" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.5:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *rt_measure_fixation* updates
            
            # if rt_measure_fixation is starting this frame...
            if rt_measure_fixation.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                rt_measure_fixation.frameNStart = frameN  # exact frame index
                rt_measure_fixation.tStart = t  # local t and not account for scr refresh
                rt_measure_fixation.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(rt_measure_fixation, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'rt_measure_fixation.started')
                # update status
                rt_measure_fixation.status = STARTED
                rt_measure_fixation.setAutoDraw(True)
            
            # if rt_measure_fixation is active this frame...
            if rt_measure_fixation.status == STARTED:
                # update params
                pass
            
            # if rt_measure_fixation is stopping this frame...
            if rt_measure_fixation.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > rt_measure_fixation.tStartRefresh + 1.5-frameTolerance:
                    # keep track of stop time/frame for later
                    rt_measure_fixation.tStop = t  # not accounting for scr refresh
                    rt_measure_fixation.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'rt_measure_fixation.stopped')
                    # update status
                    rt_measure_fixation.status = FINISHED
                    rt_measure_fixation.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in RT_measure_fixationComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "RT_measure_fixation" ---
        for thisComponent in RT_measure_fixationComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('RT_measure_fixation.stopped', globalClock.getTime())
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.500000)
        
        # --- Prepare to start Routine "RT_measure_target" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('RT_measure_target.started', globalClock.getTime())
        RT_target_fig.setPos(rt_loc)
        RT_measure_response.keys = []
        RT_measure_response.rt = []
        _RT_measure_response_allKeys = []
        # keep track of which components have finished
        RT_measure_targetComponents = [RT_target_fig, RT_measure_response]
        for thisComponent in RT_measure_targetComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "RT_measure_target" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *RT_target_fig* updates
            
            # if RT_target_fig is starting this frame...
            if RT_target_fig.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                RT_target_fig.frameNStart = frameN  # exact frame index
                RT_target_fig.tStart = t  # local t and not account for scr refresh
                RT_target_fig.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(RT_target_fig, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'RT_target_fig.started')
                # update status
                RT_target_fig.status = STARTED
                RT_target_fig.setAutoDraw(True)
            
            # if RT_target_fig is active this frame...
            if RT_target_fig.status == STARTED:
                # update params
                pass
            
            # *RT_measure_response* updates
            waitOnFlip = False
            
            # if RT_measure_response is starting this frame...
            if RT_measure_response.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                RT_measure_response.frameNStart = frameN  # exact frame index
                RT_measure_response.tStart = t  # local t and not account for scr refresh
                RT_measure_response.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(RT_measure_response, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'RT_measure_response.started')
                # update status
                RT_measure_response.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(RT_measure_response.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(RT_measure_response.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if RT_measure_response is stopping this frame...
            if RT_measure_response.status == STARTED:
                if bool(0):
                    # keep track of stop time/frame for later
                    RT_measure_response.tStop = t  # not accounting for scr refresh
                    RT_measure_response.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'RT_measure_response.stopped')
                    # update status
                    RT_measure_response.status = FINISHED
                    RT_measure_response.status = FINISHED
            if RT_measure_response.status == STARTED and not waitOnFlip:
                theseKeys = RT_measure_response.getKeys(keyList=['left','right'], ignoreKeys=["escape"], waitRelease=False)
                _RT_measure_response_allKeys.extend(theseKeys)
                if len(_RT_measure_response_allKeys):
                    RT_measure_response.keys = _RT_measure_response_allKeys[0].name  # just the first key pressed
                    RT_measure_response.rt = _RT_measure_response_allKeys[0].rt
                    RT_measure_response.duration = _RT_measure_response_allKeys[0].duration
                    # was this correct?
                    if (RT_measure_response.keys == str(rt_corr_press)) or (RT_measure_response.keys == rt_corr_press):
                        RT_measure_response.corr = 1
                    else:
                        RT_measure_response.corr = 0
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in RT_measure_targetComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "RT_measure_target" ---
        for thisComponent in RT_measure_targetComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('RT_measure_target.stopped', globalClock.getTime())
        # check responses
        if RT_measure_response.keys in ['', [], None]:  # No response was made
            RT_measure_response.keys = None
            # was no response the correct answer?!
            if str(rt_corr_press).lower() == 'none':
               RT_measure_response.corr = 1;  # correct non-response
            else:
               RT_measure_response.corr = 0;  # failed to respond (incorrectly)
        # store data for RT_measure_loop (TrialHandler)
        RT_measure_loop.addData('RT_measure_response.keys',RT_measure_response.keys)
        RT_measure_loop.addData('RT_measure_response.corr', RT_measure_response.corr)
        if RT_measure_response.keys != None:  # we had a response
            RT_measure_loop.addData('RT_measure_response.rt', RT_measure_response.rt)
            RT_measure_loop.addData('RT_measure_response.duration', RT_measure_response.duration)
        # Run 'End Routine' code from add_RT_measure_code
        rt_list.append(RT_measure_response.rt)
        
        thisExp.addData('rt_list', rt_list) 
        
        # the Routine "RT_measure_target" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 5.0 repeats of 'RT_measure_loop'
    
    
    # --- Prepare to start Routine "RT_measure_end" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('RT_measure_end.started', globalClock.getTime())
    # Run 'Begin Routine' code from RT_measuring_code
    # turn the list into a numpy array that supports 
    # mathematical operations better:
    import numpy as np
    rt_array = np.array(rt_list)
    mean_rt = rt_array.mean()
    sd_rt = rt_array.std()
    
    resp_time = mean_rt + sd_rt
    print(resp_time)
    # keep track of which components have finished
    RT_measure_endComponents = [RT_measure_endtext]
    for thisComponent in RT_measure_endComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "RT_measure_end" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 1.5:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *RT_measure_endtext* updates
        
        # if RT_measure_endtext is starting this frame...
        if RT_measure_endtext.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            RT_measure_endtext.frameNStart = frameN  # exact frame index
            RT_measure_endtext.tStart = t  # local t and not account for scr refresh
            RT_measure_endtext.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(RT_measure_endtext, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'RT_measure_endtext.started')
            # update status
            RT_measure_endtext.status = STARTED
            RT_measure_endtext.setAutoDraw(True)
        
        # if RT_measure_endtext is active this frame...
        if RT_measure_endtext.status == STARTED:
            # update params
            pass
        
        # if RT_measure_endtext is stopping this frame...
        if RT_measure_endtext.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > RT_measure_endtext.tStartRefresh + 1.5-frameTolerance:
                # keep track of stop time/frame for later
                RT_measure_endtext.tStop = t  # not accounting for scr refresh
                RT_measure_endtext.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'RT_measure_endtext.stopped')
                # update status
                RT_measure_endtext.status = FINISHED
                RT_measure_endtext.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in RT_measure_endComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "RT_measure_end" ---
    for thisComponent in RT_measure_endComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('RT_measure_end.stopped', globalClock.getTime())
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-1.500000)
    
    # --- Prepare to start Routine "StartScreen_training" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('StartScreen_training.started', globalClock.getTime())
    StartScreen_ButtonPress_training.keys = []
    StartScreen_ButtonPress_training.rt = []
    _StartScreen_ButtonPress_training_allKeys = []
    # keep track of which components have finished
    StartScreen_trainingComponents = [StartScreen_ButtonPress_training, StartScreen_text_training]
    for thisComponent in StartScreen_trainingComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "StartScreen_training" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *StartScreen_ButtonPress_training* updates
        waitOnFlip = False
        
        # if StartScreen_ButtonPress_training is starting this frame...
        if StartScreen_ButtonPress_training.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            StartScreen_ButtonPress_training.frameNStart = frameN  # exact frame index
            StartScreen_ButtonPress_training.tStart = t  # local t and not account for scr refresh
            StartScreen_ButtonPress_training.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(StartScreen_ButtonPress_training, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'StartScreen_ButtonPress_training.started')
            # update status
            StartScreen_ButtonPress_training.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(StartScreen_ButtonPress_training.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(StartScreen_ButtonPress_training.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if StartScreen_ButtonPress_training.status == STARTED and not waitOnFlip:
            theseKeys = StartScreen_ButtonPress_training.getKeys(keyList=['right'], ignoreKeys=["escape"], waitRelease=False)
            _StartScreen_ButtonPress_training_allKeys.extend(theseKeys)
            if len(_StartScreen_ButtonPress_training_allKeys):
                StartScreen_ButtonPress_training.keys = _StartScreen_ButtonPress_training_allKeys[-1].name  # just the last key pressed
                StartScreen_ButtonPress_training.rt = _StartScreen_ButtonPress_training_allKeys[-1].rt
                StartScreen_ButtonPress_training.duration = _StartScreen_ButtonPress_training_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *StartScreen_text_training* updates
        
        # if StartScreen_text_training is starting this frame...
        if StartScreen_text_training.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            StartScreen_text_training.frameNStart = frameN  # exact frame index
            StartScreen_text_training.tStart = t  # local t and not account for scr refresh
            StartScreen_text_training.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(StartScreen_text_training, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'StartScreen_text_training.started')
            # update status
            StartScreen_text_training.status = STARTED
            StartScreen_text_training.setAutoDraw(True)
        
        # if StartScreen_text_training is active this frame...
        if StartScreen_text_training.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in StartScreen_trainingComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "StartScreen_training" ---
    for thisComponent in StartScreen_trainingComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('StartScreen_training.stopped', globalClock.getTime())
    # check responses
    if StartScreen_ButtonPress_training.keys in ['', [], None]:  # No response was made
        StartScreen_ButtonPress_training.keys = None
    thisExp.addData('StartScreen_ButtonPress_training.keys',StartScreen_ButtonPress_training.keys)
    if StartScreen_ButtonPress_training.keys != None:  # we had a response
        thisExp.addData('StartScreen_ButtonPress_training.rt', StartScreen_ButtonPress_training.rt)
        thisExp.addData('StartScreen_ButtonPress_training.duration', StartScreen_ButtonPress_training.duration)
    thisExp.nextEntry()
    # the Routine "StartScreen_training" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "initial_iti_training" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('initial_iti_training.started', globalClock.getTime())
    # keep track of which components have finished
    initial_iti_trainingComponents = [initial_iti_fig_training]
    for thisComponent in initial_iti_trainingComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "initial_iti_training" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 0.5:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *initial_iti_fig_training* updates
        
        # if initial_iti_fig_training is starting this frame...
        if initial_iti_fig_training.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            initial_iti_fig_training.frameNStart = frameN  # exact frame index
            initial_iti_fig_training.tStart = t  # local t and not account for scr refresh
            initial_iti_fig_training.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(initial_iti_fig_training, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'initial_iti_fig_training.started')
            # update status
            initial_iti_fig_training.status = STARTED
            initial_iti_fig_training.setAutoDraw(True)
        
        # if initial_iti_fig_training is active this frame...
        if initial_iti_fig_training.status == STARTED:
            # update params
            pass
        
        # if initial_iti_fig_training is stopping this frame...
        if initial_iti_fig_training.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > initial_iti_fig_training.tStartRefresh + 0.5-frameTolerance:
                # keep track of stop time/frame for later
                initial_iti_fig_training.tStop = t  # not accounting for scr refresh
                initial_iti_fig_training.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'initial_iti_fig_training.stopped')
                # update status
                initial_iti_fig_training.status = FINISHED
                initial_iti_fig_training.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in initial_iti_trainingComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "initial_iti_training" ---
    for thisComponent in initial_iti_trainingComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('initial_iti_training.stopped', globalClock.getTime())
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-0.500000)
    
    # set up handler to look after randomisation of conditions etc
    PracticeLoop = data.TrialHandler(nReps=2.0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('MID_conditions.xlsx'),
        seed=None, name='PracticeLoop')
    thisExp.addLoop(PracticeLoop)  # add the loop to the experiment
    thisPracticeLoop = PracticeLoop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisPracticeLoop.rgb)
    if thisPracticeLoop != None:
        for paramName in thisPracticeLoop:
            globals()[paramName] = thisPracticeLoop[paramName]
    
    for thisPracticeLoop in PracticeLoop:
        currentLoop = PracticeLoop
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisPracticeLoop.rgb)
        if thisPracticeLoop != None:
            for paramName in thisPracticeLoop:
                globals()[paramName] = thisPracticeLoop[paramName]
        
        # --- Prepare to start Routine "CuePresentation_training" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('CuePresentation_training.started', globalClock.getTime())
        CueCircle_training.setFillColor(colour)
        CueCircle_training.setPos(location)
        CueCircle_training.setLineColor(colour)
        EarlyPressCue_training.keys = []
        EarlyPressCue_training.rt = []
        _EarlyPressCue_training_allKeys = []
        # keep track of which components have finished
        CuePresentation_trainingComponents = [CueCircle_training, EarlyPressCue_training]
        for thisComponent in CuePresentation_trainingComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "CuePresentation_training" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.25:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *CueCircle_training* updates
            
            # if CueCircle_training is starting this frame...
            if CueCircle_training.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                CueCircle_training.frameNStart = frameN  # exact frame index
                CueCircle_training.tStart = t  # local t and not account for scr refresh
                CueCircle_training.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(CueCircle_training, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'CueCircle_training.started')
                # update status
                CueCircle_training.status = STARTED
                CueCircle_training.setAutoDraw(True)
            
            # if CueCircle_training is active this frame...
            if CueCircle_training.status == STARTED:
                # update params
                pass
            
            # if CueCircle_training is stopping this frame...
            if CueCircle_training.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > CueCircle_training.tStartRefresh + 0.25-frameTolerance:
                    # keep track of stop time/frame for later
                    CueCircle_training.tStop = t  # not accounting for scr refresh
                    CueCircle_training.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'CueCircle_training.stopped')
                    # update status
                    CueCircle_training.status = FINISHED
                    CueCircle_training.setAutoDraw(False)
            
            # *EarlyPressCue_training* updates
            waitOnFlip = False
            
            # if EarlyPressCue_training is starting this frame...
            if EarlyPressCue_training.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                EarlyPressCue_training.frameNStart = frameN  # exact frame index
                EarlyPressCue_training.tStart = t  # local t and not account for scr refresh
                EarlyPressCue_training.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(EarlyPressCue_training, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'EarlyPressCue_training.started')
                # update status
                EarlyPressCue_training.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(EarlyPressCue_training.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(EarlyPressCue_training.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if EarlyPressCue_training is stopping this frame...
            if EarlyPressCue_training.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > EarlyPressCue_training.tStartRefresh + 0.25-frameTolerance:
                    # keep track of stop time/frame for later
                    EarlyPressCue_training.tStop = t  # not accounting for scr refresh
                    EarlyPressCue_training.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'EarlyPressCue_training.stopped')
                    # update status
                    EarlyPressCue_training.status = FINISHED
                    EarlyPressCue_training.status = FINISHED
            if EarlyPressCue_training.status == STARTED and not waitOnFlip:
                theseKeys = EarlyPressCue_training.getKeys(keyList=['left', 'right'], ignoreKeys=["escape"], waitRelease=False)
                _EarlyPressCue_training_allKeys.extend(theseKeys)
                if len(_EarlyPressCue_training_allKeys):
                    EarlyPressCue_training.keys = _EarlyPressCue_training_allKeys[0].name  # just the first key pressed
                    EarlyPressCue_training.rt = _EarlyPressCue_training_allKeys[0].rt
                    EarlyPressCue_training.duration = _EarlyPressCue_training_allKeys[0].duration
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in CuePresentation_trainingComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "CuePresentation_training" ---
        for thisComponent in CuePresentation_trainingComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('CuePresentation_training.stopped', globalClock.getTime())
        # check responses
        if EarlyPressCue_training.keys in ['', [], None]:  # No response was made
            EarlyPressCue_training.keys = None
        PracticeLoop.addData('EarlyPressCue_training.keys',EarlyPressCue_training.keys)
        if EarlyPressCue_training.keys != None:  # we had a response
            PracticeLoop.addData('EarlyPressCue_training.rt', EarlyPressCue_training.rt)
            PracticeLoop.addData('EarlyPressCue_training.duration', EarlyPressCue_training.duration)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.250000)
        
        # --- Prepare to start Routine "Fixation2000_training" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('Fixation2000_training.started', globalClock.getTime())
        EarlyPressFixation_training.keys = []
        EarlyPressFixation_training.rt = []
        _EarlyPressFixation_training_allKeys = []
        # keep track of which components have finished
        Fixation2000_trainingComponents = [Fixation2000_fig_training, EarlyPressFixation_training]
        for thisComponent in Fixation2000_trainingComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Fixation2000_training" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *Fixation2000_fig_training* updates
            
            # if Fixation2000_fig_training is starting this frame...
            if Fixation2000_fig_training.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Fixation2000_fig_training.frameNStart = frameN  # exact frame index
                Fixation2000_fig_training.tStart = t  # local t and not account for scr refresh
                Fixation2000_fig_training.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Fixation2000_fig_training, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Fixation2000_fig_training.started')
                # update status
                Fixation2000_fig_training.status = STARTED
                Fixation2000_fig_training.setAutoDraw(True)
            
            # if Fixation2000_fig_training is active this frame...
            if Fixation2000_fig_training.status == STARTED:
                # update params
                pass
            
            # if Fixation2000_fig_training is stopping this frame...
            if Fixation2000_fig_training.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > Fixation2000_fig_training.tStartRefresh + 2-frameTolerance:
                    # keep track of stop time/frame for later
                    Fixation2000_fig_training.tStop = t  # not accounting for scr refresh
                    Fixation2000_fig_training.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Fixation2000_fig_training.stopped')
                    # update status
                    Fixation2000_fig_training.status = FINISHED
                    Fixation2000_fig_training.setAutoDraw(False)
            
            # *EarlyPressFixation_training* updates
            waitOnFlip = False
            
            # if EarlyPressFixation_training is starting this frame...
            if EarlyPressFixation_training.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                EarlyPressFixation_training.frameNStart = frameN  # exact frame index
                EarlyPressFixation_training.tStart = t  # local t and not account for scr refresh
                EarlyPressFixation_training.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(EarlyPressFixation_training, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'EarlyPressFixation_training.started')
                # update status
                EarlyPressFixation_training.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(EarlyPressFixation_training.clock.reset)  # t=0 on next screen flip
            
            # if EarlyPressFixation_training is stopping this frame...
            if EarlyPressFixation_training.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > EarlyPressFixation_training.tStartRefresh + 2.0-frameTolerance:
                    # keep track of stop time/frame for later
                    EarlyPressFixation_training.tStop = t  # not accounting for scr refresh
                    EarlyPressFixation_training.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'EarlyPressFixation_training.stopped')
                    # update status
                    EarlyPressFixation_training.status = FINISHED
                    EarlyPressFixation_training.status = FINISHED
            if EarlyPressFixation_training.status == STARTED and not waitOnFlip:
                theseKeys = EarlyPressFixation_training.getKeys(keyList=['left','right'], ignoreKeys=["escape"], waitRelease=False)
                _EarlyPressFixation_training_allKeys.extend(theseKeys)
                if len(_EarlyPressFixation_training_allKeys):
                    EarlyPressFixation_training.keys = _EarlyPressFixation_training_allKeys[0].name  # just the first key pressed
                    EarlyPressFixation_training.rt = _EarlyPressFixation_training_allKeys[0].rt
                    EarlyPressFixation_training.duration = _EarlyPressFixation_training_allKeys[0].duration
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in Fixation2000_trainingComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Fixation2000_training" ---
        for thisComponent in Fixation2000_trainingComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('Fixation2000_training.stopped', globalClock.getTime())
        # check responses
        if EarlyPressFixation_training.keys in ['', [], None]:  # No response was made
            EarlyPressFixation_training.keys = None
        PracticeLoop.addData('EarlyPressFixation_training.keys',EarlyPressFixation_training.keys)
        if EarlyPressFixation_training.keys != None:  # we had a response
            PracticeLoop.addData('EarlyPressFixation_training.rt', EarlyPressFixation_training.rt)
            PracticeLoop.addData('EarlyPressFixation_training.duration', EarlyPressFixation_training.duration)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.000000)
        
        # --- Prepare to start Routine "TargetPresentation_training" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('TargetPresentation_training.started', globalClock.getTime())
        # Run 'Begin Routine' code from TargetPresTiming_code_training
        # Calculates target presentation time, based on performance of last 6 trials, and
        # then on last 10 trials once that number of trials has been reached. 
        # For last 6: 
        # If accuracy on last 6 is greater than 66%, target is presented 20ms shorter.
        # If accuracy on last 6 is less than 66%, target is presented 50ms longer.
        # For last 10: 
        # If accuracy on last 10 is greater than 60%, target is presented 20ms shorter.
        # If accuracy on last 10 is less than 60%, target is presented 50ms longer.
        # If accuracy on last 10 is 60%, target presentation time does not change. 
        if trial_num <= 4:
            resp_time = resp_time;
            thisExp.addData('full_target_time', resp_time);
        elif trial_num > 5 and trial_num <= 8:
            last_6_acc = calibration_accuracy[-6:];
            acc_ratio = sum(last_6_acc) / len(last_6_acc);
            thisExp.addData('acc_ratio', acc_ratio);
            if acc_ratio <= 0.66: #target presented for 50 ms longer
                resp_time = resp_time + 0.05
                thisExp.addData('full_target_time', resp_time);
            else: #target presented for 20ms shorter
                resp_time = resp_time - 0.02
                thisExp.addData('full_target_time', resp_time);
        else:
            last_10_acc = calibration_accuracy[-10:];
            acc_ratio = sum(last_10_acc) / len(last_10_acc);
            thisExp.addData('acc_ratio', acc_ratio);
            if acc_ratio < 0.60: #target presented for 50 ms longer
                resp_time = resp_time + 0.05
                thisExp.addData('full_target_time', resp_time);
            elif acc_ratio == 0.60: #target presentation time stays the same. 
                resp_time = resp_time
                thisExp.addData('full_target_time', resp_time);
            else: #target presented for 20ms shorter
                resp_time = resp_time - 0.02
                thisExp.addData('full_target_time', resp_time);
        ButtonPressTarget_training.keys = []
        ButtonPressTarget_training.rt = []
        _ButtonPressTarget_training_allKeys = []
        # keep track of which components have finished
        TargetPresentation_trainingComponents = [TargetPresentation_fig_training, ButtonPressTarget_training]
        for thisComponent in TargetPresentation_trainingComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "TargetPresentation_training" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *TargetPresentation_fig_training* updates
            
            # if TargetPresentation_fig_training is starting this frame...
            if TargetPresentation_fig_training.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                TargetPresentation_fig_training.frameNStart = frameN  # exact frame index
                TargetPresentation_fig_training.tStart = t  # local t and not account for scr refresh
                TargetPresentation_fig_training.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(TargetPresentation_fig_training, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'TargetPresentation_fig_training.started')
                # update status
                TargetPresentation_fig_training.status = STARTED
                TargetPresentation_fig_training.setAutoDraw(True)
            
            # if TargetPresentation_fig_training is active this frame...
            if TargetPresentation_fig_training.status == STARTED:
                # update params
                pass
            
            # if TargetPresentation_fig_training is stopping this frame...
            if TargetPresentation_fig_training.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > TargetPresentation_fig_training.tStartRefresh + resp_time-frameTolerance:
                    # keep track of stop time/frame for later
                    TargetPresentation_fig_training.tStop = t  # not accounting for scr refresh
                    TargetPresentation_fig_training.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'TargetPresentation_fig_training.stopped')
                    # update status
                    TargetPresentation_fig_training.status = FINISHED
                    TargetPresentation_fig_training.setAutoDraw(False)
            
            # *ButtonPressTarget_training* updates
            waitOnFlip = False
            
            # if ButtonPressTarget_training is starting this frame...
            if ButtonPressTarget_training.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                ButtonPressTarget_training.frameNStart = frameN  # exact frame index
                ButtonPressTarget_training.tStart = t  # local t and not account for scr refresh
                ButtonPressTarget_training.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(ButtonPressTarget_training, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'ButtonPressTarget_training.started')
                # update status
                ButtonPressTarget_training.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(ButtonPressTarget_training.clock.reset)  # t=0 on next screen flip
            
            # if ButtonPressTarget_training is stopping this frame...
            if ButtonPressTarget_training.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > ButtonPressTarget_training.tStartRefresh + resp_time-frameTolerance:
                    # keep track of stop time/frame for later
                    ButtonPressTarget_training.tStop = t  # not accounting for scr refresh
                    ButtonPressTarget_training.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'ButtonPressTarget_training.stopped')
                    # update status
                    ButtonPressTarget_training.status = FINISHED
                    ButtonPressTarget_training.status = FINISHED
            if ButtonPressTarget_training.status == STARTED and not waitOnFlip:
                theseKeys = ButtonPressTarget_training.getKeys(keyList=['left','right'], ignoreKeys=["escape"], waitRelease=False)
                _ButtonPressTarget_training_allKeys.extend(theseKeys)
                if len(_ButtonPressTarget_training_allKeys):
                    ButtonPressTarget_training.keys = _ButtonPressTarget_training_allKeys[0].name  # just the first key pressed
                    ButtonPressTarget_training.rt = _ButtonPressTarget_training_allKeys[0].rt
                    ButtonPressTarget_training.duration = _ButtonPressTarget_training_allKeys[0].duration
                    # was this correct?
                    if (ButtonPressTarget_training.keys == str(corr_button)) or (ButtonPressTarget_training.keys == corr_button):
                        ButtonPressTarget_training.corr = 1
                    else:
                        ButtonPressTarget_training.corr = 0
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in TargetPresentation_trainingComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "TargetPresentation_training" ---
        for thisComponent in TargetPresentation_trainingComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('TargetPresentation_training.stopped', globalClock.getTime())
        # check responses
        if ButtonPressTarget_training.keys in ['', [], None]:  # No response was made
            ButtonPressTarget_training.keys = None
            # was no response the correct answer?!
            if str(corr_button).lower() == 'none':
               ButtonPressTarget_training.corr = 1;  # correct non-response
            else:
               ButtonPressTarget_training.corr = 0;  # failed to respond (incorrectly)
        # store data for PracticeLoop (TrialHandler)
        PracticeLoop.addData('ButtonPressTarget_training.keys',ButtonPressTarget_training.keys)
        PracticeLoop.addData('ButtonPressTarget_training.corr', ButtonPressTarget_training.corr)
        if ButtonPressTarget_training.keys != None:  # we had a response
            PracticeLoop.addData('ButtonPressTarget_training.rt', ButtonPressTarget_training.rt)
            PracticeLoop.addData('ButtonPressTarget_training.duration', ButtonPressTarget_training.duration)
        # Run 'End Routine' code from Feedbacksaving_code_training
        feedbackver_training = [];
        # Prepares which feedback version is shown, based on win or loss cues and button press performance. 
        if EarlyPressCue_training.keys != None or EarlyPressFixation_training.keys != None and colour == "blue":
            feedbackver_training = "3"; # Early press win cue -> did not win point. 
            thisExp.addData('outcome_label_training', "Early")
            #thisExp.addData('practice_outcome_val', -2)
            calibration_accuracy.append(0)
        elif EarlyPressCue_training.keys != None or EarlyPressFixation_training.keys != None and colour == "red":
            feedbackver_training = "4"; # Early press loss cue -> lost point.
            thisExp.addData('outcome_label_training', "Early")
            #thisExp.addData('practice_outcome_val', -2)
            calibration_accuracy.append(0)
        elif EarlyPressCue_training.keys != None or EarlyPressFixation_training.keys != None and colour == "yellow":
            feedbackver_training = "6"; # Early press neutral cue -> points stay the same but incorrect.
            thisExp.addData('outcome_label_training', "Early")
            #thisExp.addData('practice_outcome_val', -2)
            calibration_accuracy.append(0)
        elif ButtonPressTarget_training.corr == 1 and colour == "blue":
            feedbackver_training = "1"; # Correct press win cue -> won point.
            thisExp.addData('outcome_label_training', "Correct")
            #thisExp.addData('practice_outcome_val', 1)
            calibration_accuracy.append(1)
        elif ButtonPressTarget_training.corr == 1 and colour == "red":
            feedbackver_training = "2"; # Correct press loss cue -> did not lose point. 
            thisExp.addData('outcome_label_training', "Correct")
            #thisExp.addData('practice_outcome_val', 1)
            calibration_accuracy.append(1)
        elif ButtonPressTarget_training.corr == 1 and colour == "yellow":
            feedbackver_training = "5"; # Correct press neutral cue -> points stay the same. 
            thisExp.addData('outcome_label_training', "Correct")
            #thisExp.addData('practice_outcome_val', 1)
            calibration_accuracy.append(1)
        elif ButtonPressTarget_training.corr == 0 and colour == "blue":
            feedbackver_training = "3"; # Incorrect press win cue -> did not win point.
            thisExp.addData('outcome_label_training', "Incorrect")
            #thisExp.addData('practice_outcome_val', -1)
            calibration_accuracy.append(0)
        elif ButtonPressTarget_training.corr == 0 and colour == "yellow":
            feedbackver_training = "6"; # Incorrect press neutral cue -> points stay the same.
            thisExp.addData('outcome_label_training', "Incorrect")
            #thisExp.addData('practice_outcome_val', -1)
            calibration_accuracy.append(0)
        else:
            feedbackver_training = "4"; # Incorrect press loss cue -> lost point.
            thisExp.addData('outcome_label_training', "Incorrect")
            #thisExp.addData('practice_outcome_val', -1)
            calibration_accuracy.append(0)
            
        # the Routine "TargetPresentation_training" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "FeedbackCode_training" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('FeedbackCode_training.started', globalClock.getTime())
        # Run 'Begin Routine' code from practice_feedbacktextcode
        # Correct press win cue (blue)
        if feedbackver_training == "1":
            text_training = "Sie haben einen Punkt gewonnen";
            textcolour_training = 'green';
            reward_counter_training += 1;
            
        # Correct press loss cue (red)
        elif feedbackver_training == "2":
            text_training = "Sie haben keinen Punkt verloren";
            textcolour_training = 'green';
            
        # Incorrect press win cue (blue)    
        elif feedbackver_training == "3":
            text_training = "Sie haben keinen Punkt gewonnen";
            textcolour_training = 'red';
            
        # Incorrect press loss cue (red)        
        elif feedbackver_training == "4":
            text_training = "Sie haben einen Punkt verloren";
            textcolour_training = 'red';
            reward_counter_training -= 1;
        
        # Neutral cue, correct press, points unchanged (yellow)
        elif feedbackver_training == "5":
            text_training = "Sie haben keinen Punkt verloren";
            textcolour_training = 'green';
            
        # Neutral cue, incorrect press, points unchanged (yellow)
        else: #Feedback version 6
            text_training = "Sie haben keinen Punkt verloren";
            textcolour_training = 'red';
           
        
        
        text_Feedback_training.setColor(textcolour_training, colorSpace='rgb')
        text_Feedback_training.setText(text_training)
        text_treatCounter_training.setText(reward_counter_training)
        # keep track of which components have finished
        FeedbackCode_trainingComponents = [text_Feedback_training, text_treatCounter_training]
        for thisComponent in FeedbackCode_trainingComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "FeedbackCode_training" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # is it time to end the Routine? (based on local clock)
            if tThisFlip > 1-frameTolerance:
                continueRoutine = False
            
            # *text_Feedback_training* updates
            
            # if text_Feedback_training is starting this frame...
            if text_Feedback_training.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                text_Feedback_training.frameNStart = frameN  # exact frame index
                text_Feedback_training.tStart = t  # local t and not account for scr refresh
                text_Feedback_training.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_Feedback_training, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_Feedback_training.started')
                # update status
                text_Feedback_training.status = STARTED
                text_Feedback_training.setAutoDraw(True)
            
            # if text_Feedback_training is active this frame...
            if text_Feedback_training.status == STARTED:
                # update params
                pass
            
            # if text_Feedback_training is stopping this frame...
            if text_Feedback_training.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_Feedback_training.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    text_Feedback_training.tStop = t  # not accounting for scr refresh
                    text_Feedback_training.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_Feedback_training.stopped')
                    # update status
                    text_Feedback_training.status = FINISHED
                    text_Feedback_training.setAutoDraw(False)
            
            # *text_treatCounter_training* updates
            
            # if text_treatCounter_training is starting this frame...
            if text_treatCounter_training.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                text_treatCounter_training.frameNStart = frameN  # exact frame index
                text_treatCounter_training.tStart = t  # local t and not account for scr refresh
                text_treatCounter_training.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_treatCounter_training, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_treatCounter_training.started')
                # update status
                text_treatCounter_training.status = STARTED
                text_treatCounter_training.setAutoDraw(True)
            
            # if text_treatCounter_training is active this frame...
            if text_treatCounter_training.status == STARTED:
                # update params
                pass
            
            # if text_treatCounter_training is stopping this frame...
            if text_treatCounter_training.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_treatCounter_training.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    text_treatCounter_training.tStop = t  # not accounting for scr refresh
                    text_treatCounter_training.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_treatCounter_training.stopped')
                    # update status
                    text_treatCounter_training.status = FINISHED
                    text_treatCounter_training.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in FeedbackCode_trainingComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "FeedbackCode_training" ---
        for thisComponent in FeedbackCode_trainingComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('FeedbackCode_training.stopped', globalClock.getTime())
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        
        # --- Prepare to start Routine "ITI500_training" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('ITI500_training.started', globalClock.getTime())
        # keep track of which components have finished
        ITI500_trainingComponents = [trial_ITI_training]
        for thisComponent in ITI500_trainingComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "ITI500_training" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.5:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *trial_ITI_training* updates
            
            # if trial_ITI_training is starting this frame...
            if trial_ITI_training.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                trial_ITI_training.frameNStart = frameN  # exact frame index
                trial_ITI_training.tStart = t  # local t and not account for scr refresh
                trial_ITI_training.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(trial_ITI_training, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'trial_ITI_training.started')
                # update status
                trial_ITI_training.status = STARTED
                trial_ITI_training.setAutoDraw(True)
            
            # if trial_ITI_training is active this frame...
            if trial_ITI_training.status == STARTED:
                # update params
                pass
            
            # if trial_ITI_training is stopping this frame...
            if trial_ITI_training.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > trial_ITI_training.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    trial_ITI_training.tStop = t  # not accounting for scr refresh
                    trial_ITI_training.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'trial_ITI_training.stopped')
                    # update status
                    trial_ITI_training.status = FINISHED
                    trial_ITI_training.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in ITI500_trainingComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "ITI500_training" ---
        for thisComponent in ITI500_trainingComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('ITI500_training.stopped', globalClock.getTime())
        # Run 'End Routine' code from trial_num_code_training
        # Trial_num is used in code_TargetPresTiming
        trial_num = trial_num + 1;
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.500000)
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 2.0 repeats of 'PracticeLoop'
    
    
    # --- Prepare to start Routine "EndScreen_training" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('EndScreen_training.started', globalClock.getTime())
    endTraining_startMain.keys = []
    endTraining_startMain.rt = []
    _endTraining_startMain_allKeys = []
    # keep track of which components have finished
    EndScreen_trainingComponents = [EndScreenText_training, endTraining_startMain]
    for thisComponent in EndScreen_trainingComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "EndScreen_training" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *EndScreenText_training* updates
        
        # if EndScreenText_training is starting this frame...
        if EndScreenText_training.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            EndScreenText_training.frameNStart = frameN  # exact frame index
            EndScreenText_training.tStart = t  # local t and not account for scr refresh
            EndScreenText_training.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(EndScreenText_training, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'EndScreenText_training.started')
            # update status
            EndScreenText_training.status = STARTED
            EndScreenText_training.setAutoDraw(True)
        
        # if EndScreenText_training is active this frame...
        if EndScreenText_training.status == STARTED:
            # update params
            pass
        
        # *endTraining_startMain* updates
        waitOnFlip = False
        
        # if endTraining_startMain is starting this frame...
        if endTraining_startMain.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            endTraining_startMain.frameNStart = frameN  # exact frame index
            endTraining_startMain.tStart = t  # local t and not account for scr refresh
            endTraining_startMain.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(endTraining_startMain, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'endTraining_startMain.started')
            # update status
            endTraining_startMain.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(endTraining_startMain.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(endTraining_startMain.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if endTraining_startMain.status == STARTED and not waitOnFlip:
            theseKeys = endTraining_startMain.getKeys(keyList=['s'], ignoreKeys=["escape"], waitRelease=False)
            _endTraining_startMain_allKeys.extend(theseKeys)
            if len(_endTraining_startMain_allKeys):
                endTraining_startMain.keys = _endTraining_startMain_allKeys[0].name  # just the first key pressed
                endTraining_startMain.rt = _endTraining_startMain_allKeys[0].rt
                endTraining_startMain.duration = _endTraining_startMain_allKeys[0].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in EndScreen_trainingComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "EndScreen_training" ---
    for thisComponent in EndScreen_trainingComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('EndScreen_training.stopped', globalClock.getTime())
    # check responses
    if endTraining_startMain.keys in ['', [], None]:  # No response was made
        endTraining_startMain.keys = None
    thisExp.addData('endTraining_startMain.keys',endTraining_startMain.keys)
    if endTraining_startMain.keys != None:  # we had a response
        thisExp.addData('endTraining_startMain.rt', endTraining_startMain.rt)
        thisExp.addData('endTraining_startMain.duration', endTraining_startMain.duration)
    thisExp.nextEntry()
    # the Routine "EndScreen_training" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "WelcomeScreen" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('WelcomeScreen.started', globalClock.getTime())
    StartScreen_ButtonPress.keys = []
    StartScreen_ButtonPress.rt = []
    _StartScreen_ButtonPress_allKeys = []
    # keep track of which components have finished
    WelcomeScreenComponents = [StartScreen_ButtonPress, textStartScreen]
    for thisComponent in WelcomeScreenComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "WelcomeScreen" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *StartScreen_ButtonPress* updates
        waitOnFlip = False
        
        # if StartScreen_ButtonPress is starting this frame...
        if StartScreen_ButtonPress.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            StartScreen_ButtonPress.frameNStart = frameN  # exact frame index
            StartScreen_ButtonPress.tStart = t  # local t and not account for scr refresh
            StartScreen_ButtonPress.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(StartScreen_ButtonPress, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'StartScreen_ButtonPress.started')
            # update status
            StartScreen_ButtonPress.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(StartScreen_ButtonPress.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(StartScreen_ButtonPress.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if StartScreen_ButtonPress.status == STARTED and not waitOnFlip:
            theseKeys = StartScreen_ButtonPress.getKeys(keyList=['right'], ignoreKeys=["escape"], waitRelease=False)
            _StartScreen_ButtonPress_allKeys.extend(theseKeys)
            if len(_StartScreen_ButtonPress_allKeys):
                StartScreen_ButtonPress.keys = _StartScreen_ButtonPress_allKeys[-1].name  # just the last key pressed
                StartScreen_ButtonPress.rt = _StartScreen_ButtonPress_allKeys[-1].rt
                StartScreen_ButtonPress.duration = _StartScreen_ButtonPress_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *textStartScreen* updates
        
        # if textStartScreen is starting this frame...
        if textStartScreen.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            textStartScreen.frameNStart = frameN  # exact frame index
            textStartScreen.tStart = t  # local t and not account for scr refresh
            textStartScreen.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(textStartScreen, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'textStartScreen.started')
            # update status
            textStartScreen.status = STARTED
            textStartScreen.setAutoDraw(True)
        
        # if textStartScreen is active this frame...
        if textStartScreen.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in WelcomeScreenComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "WelcomeScreen" ---
    for thisComponent in WelcomeScreenComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('WelcomeScreen.stopped', globalClock.getTime())
    # check responses
    if StartScreen_ButtonPress.keys in ['', [], None]:  # No response was made
        StartScreen_ButtonPress.keys = None
    thisExp.addData('StartScreen_ButtonPress.keys',StartScreen_ButtonPress.keys)
    if StartScreen_ButtonPress.keys != None:  # we had a response
        thisExp.addData('StartScreen_ButtonPress.rt', StartScreen_ButtonPress.rt)
        thisExp.addData('StartScreen_ButtonPress.duration', StartScreen_ButtonPress.duration)
    thisExp.nextEntry()
    # the Routine "WelcomeScreen" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "initial_iti" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('initial_iti.started', globalClock.getTime())
    # keep track of which components have finished
    initial_itiComponents = [first_ITI_fig]
    for thisComponent in initial_itiComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "initial_iti" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 0.5:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *first_ITI_fig* updates
        
        # if first_ITI_fig is starting this frame...
        if first_ITI_fig.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            first_ITI_fig.frameNStart = frameN  # exact frame index
            first_ITI_fig.tStart = t  # local t and not account for scr refresh
            first_ITI_fig.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(first_ITI_fig, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'first_ITI_fig.started')
            # update status
            first_ITI_fig.status = STARTED
            first_ITI_fig.setAutoDraw(True)
        
        # if first_ITI_fig is active this frame...
        if first_ITI_fig.status == STARTED:
            # update params
            pass
        
        # if first_ITI_fig is stopping this frame...
        if first_ITI_fig.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > first_ITI_fig.tStartRefresh + 0.5-frameTolerance:
                # keep track of stop time/frame for later
                first_ITI_fig.tStop = t  # not accounting for scr refresh
                first_ITI_fig.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'first_ITI_fig.stopped')
                # update status
                first_ITI_fig.status = FINISHED
                first_ITI_fig.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in initial_itiComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "initial_iti" ---
    for thisComponent in initial_itiComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('initial_iti.stopped', globalClock.getTime())
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-0.500000)
    
    # set up handler to look after randomisation of conditions etc
    MainLoop = data.TrialHandler(nReps=15.0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('MID_conditions.xlsx'),
        seed=None, name='MainLoop')
    thisExp.addLoop(MainLoop)  # add the loop to the experiment
    thisMainLoop = MainLoop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisMainLoop.rgb)
    if thisMainLoop != None:
        for paramName in thisMainLoop:
            globals()[paramName] = thisMainLoop[paramName]
    
    for thisMainLoop in MainLoop:
        currentLoop = MainLoop
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisMainLoop.rgb)
        if thisMainLoop != None:
            for paramName in thisMainLoop:
                globals()[paramName] = thisMainLoop[paramName]
        
        # --- Prepare to start Routine "CuePresentation" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('CuePresentation.started', globalClock.getTime())
        CueCircle.setFillColor(colour)
        CueCircle.setPos(location)
        CueCircle.setLineColor(colour)
        EarlyPressCue.keys = []
        EarlyPressCue.rt = []
        _EarlyPressCue_allKeys = []
        # Run 'Begin Routine' code from button_count_cue_code
        cue_button_count = 0 # Used for pushing button presses
        # keep track of which components have finished
        CuePresentationComponents = [CueCircle, EarlyPressCue]
        for thisComponent in CuePresentationComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "CuePresentation" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.25:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *CueCircle* updates
            
            # if CueCircle is starting this frame...
            if CueCircle.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:

                 # Send LSL Marker : 
                if colour == 'blue': 
                    mark = 'Win_cue'
                    outlet.push_sample([mark])  # Push event marker.
                elif colour == 'yellow': 
                    mark = 'Neutral_cue'
                    outlet.push_sample([mark])  # Push event marker.
                else: 
                    mark = 'Loss_cue'
                    outlet.push_sample([mark])  # Push event marker.

                # keep track of start time/frame for later
                CueCircle.frameNStart = frameN  # exact frame index
                CueCircle.tStart = t  # local t and not account for scr refresh
                CueCircle.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(CueCircle, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'CueCircle.started')
                # update status
                CueCircle.status = STARTED
                CueCircle.setAutoDraw(True)
            
            # if CueCircle is active this frame...
            if CueCircle.status == STARTED:
                # update params
                pass
            
            # if CueCircle is stopping this frame...
            if CueCircle.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > CueCircle.tStartRefresh + 0.25-frameTolerance:
                    # keep track of stop time/frame for later
                    CueCircle.tStop = t  # not accounting for scr refresh
                    CueCircle.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'CueCircle.stopped')
                    # update status
                    CueCircle.status = FINISHED
                    CueCircle.setAutoDraw(False)
            
            # *EarlyPressCue* updates
            waitOnFlip = False
            
            # if EarlyPressCue is starting this frame...
            if EarlyPressCue.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                EarlyPressCue.frameNStart = frameN  # exact frame index
                EarlyPressCue.tStart = t  # local t and not account for scr refresh
                EarlyPressCue.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(EarlyPressCue, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'EarlyPressCue.started')
                # update status
                EarlyPressCue.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(EarlyPressCue.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(EarlyPressCue.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if EarlyPressCue is stopping this frame...
            if EarlyPressCue.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > EarlyPressCue.tStartRefresh + 0.25-frameTolerance:
                    # keep track of stop time/frame for later
                    EarlyPressCue.tStop = t  # not accounting for scr refresh
                    EarlyPressCue.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'EarlyPressCue.stopped')
                    # update status
                    EarlyPressCue.status = FINISHED
                    EarlyPressCue.status = FINISHED
            if EarlyPressCue.status == STARTED and not waitOnFlip:
                theseKeys = EarlyPressCue.getKeys(keyList=['left', 'right'], ignoreKeys=["escape"], waitRelease=False)
                _EarlyPressCue_allKeys.extend(theseKeys)
                if len(_EarlyPressCue_allKeys):
                    EarlyPressCue.keys = _EarlyPressCue_allKeys[0].name  # just the first key pressed
                    EarlyPressCue.rt = _EarlyPressCue_allKeys[0].rt
                    EarlyPressCue.duration = _EarlyPressCue_allKeys[0].duration
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in CuePresentationComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "CuePresentation" ---
        for thisComponent in CuePresentationComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('CuePresentation.stopped', globalClock.getTime())
        # check responses
        if EarlyPressCue.keys in ['', [], None]:  # No response was made
            EarlyPressCue.keys = None
        MainLoop.addData('EarlyPressCue.keys',EarlyPressCue.keys)
        if EarlyPressCue.keys != None:  # we had a response

            # Send LSL Marker : Response as string
            mystring = ' '.join(map(str,EarlyPressCue.keys))
            resp = "Early"
            print("Response: [%s]" % resp)    
            outlet.push_sample([resp])  # Push event marker.

            MainLoop.addData('EarlyPressCue.rt', EarlyPressCue.rt)
            MainLoop.addData('EarlyPressCue.duration', EarlyPressCue.duration)
        # Run 'End Routine' code from button_count_cue_code
        if EarlyPressCue.keys != None:
            cue_button_count = 1
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.250000)
        
        # --- Prepare to start Routine "Fixation2000" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('Fixation2000.started', globalClock.getTime())
        EarlyPressFixation.keys = []
        EarlyPressFixation.rt = []
        _EarlyPressFixation_allKeys = []
        # Run 'Begin Routine' code from fix_count_button_code
        fix_button_count = 0 #to count if button pressed
        # keep track of which components have finished
        Fixation2000Components = [FixationScreen, EarlyPressFixation]
        for thisComponent in Fixation2000Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Fixation2000" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *FixationScreen* updates
            
            # if FixationScreen is starting this frame...
            if FixationScreen.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                
                # Send LSL Marker : 
                mark = 'Fixation2000'
                outlet.push_sample([mark])  # Push event marker.                    

                # keep track of start time/frame for later
                FixationScreen.frameNStart = frameN  # exact frame index
                FixationScreen.tStart = t  # local t and not account for scr refresh
                FixationScreen.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(FixationScreen, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'FixationScreen.started')
                # update status
                FixationScreen.status = STARTED
                FixationScreen.setAutoDraw(True)
            
            # if FixationScreen is active this frame...
            if FixationScreen.status == STARTED:
                # update params
                pass
            
            # if FixationScreen is stopping this frame...
            if FixationScreen.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > FixationScreen.tStartRefresh + 2-frameTolerance:
                    # keep track of stop time/frame for later
                    FixationScreen.tStop = t  # not accounting for scr refresh
                    FixationScreen.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'FixationScreen.stopped')
                    # update status
                    FixationScreen.status = FINISHED
                    FixationScreen.setAutoDraw(False)
            
            # *EarlyPressFixation* updates
            waitOnFlip = False
            
            # if EarlyPressFixation is starting this frame...
            if EarlyPressFixation.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                EarlyPressFixation.frameNStart = frameN  # exact frame index
                EarlyPressFixation.tStart = t  # local t and not account for scr refresh
                EarlyPressFixation.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(EarlyPressFixation, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'EarlyPressFixation.started')
                # update status
                EarlyPressFixation.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(EarlyPressFixation.clock.reset)  # t=0 on next screen flip
            
            # if EarlyPressFixation is stopping this frame...
            if EarlyPressFixation.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > EarlyPressFixation.tStartRefresh + 2.0-frameTolerance:
                    # keep track of stop time/frame for later
                    EarlyPressFixation.tStop = t  # not accounting for scr refresh
                    EarlyPressFixation.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'EarlyPressFixation.stopped')
                    # update status
                    EarlyPressFixation.status = FINISHED
                    EarlyPressFixation.status = FINISHED
            if EarlyPressFixation.status == STARTED and not waitOnFlip:
                theseKeys = EarlyPressFixation.getKeys(keyList=['left','right'], ignoreKeys=["escape"], waitRelease=False)
                _EarlyPressFixation_allKeys.extend(theseKeys)
                if len(_EarlyPressFixation_allKeys):
                    EarlyPressFixation.keys = _EarlyPressFixation_allKeys[0].name  # just the first key pressed
                    EarlyPressFixation.rt = _EarlyPressFixation_allKeys[0].rt
                    EarlyPressFixation.duration = _EarlyPressFixation_allKeys[0].duration
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in Fixation2000Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Fixation2000" ---
        for thisComponent in Fixation2000Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('Fixation2000.stopped', globalClock.getTime())
        # check responses
        if EarlyPressFixation.keys in ['', [], None]:  # No response was made
            EarlyPressFixation.keys = None
        MainLoop.addData('EarlyPressFixation.keys',EarlyPressFixation.keys)
        if EarlyPressFixation.keys != None:  # we had a response

            # Send LSL Marker : Response as string
            mystring = ' '.join(map(str,EarlyPressFixation.keys))
            resp = "Early"
            print("Response: [%s]" % resp)    
            outlet.push_sample([resp])  # Push event marker.

            MainLoop.addData('EarlyPressFixation.rt', EarlyPressFixation.rt)
            MainLoop.addData('EarlyPressFixation.duration', EarlyPressFixation.duration)
        # Run 'End Routine' code from fix_count_button_code
        if EarlyPressFixation.keys != None:
            fix_button_count = 1
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.000000)
        
        # --- Prepare to start Routine "TargetPresentation" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('TargetPresentation.started', globalClock.getTime())
        # Run 'Begin Routine' code from codeTargetPresTiming
        # Calculates target presentation time, based on performance of last 6 trials, and
        # then on last 10 trials once that number of trials has been reached. 
        # For last 6: 
        # If accuracy on last 6 is greater than 66%, target is presented 20ms shorter.
        # If accuracy on last 6 is less than 66%, target is presented 50ms longer.
        # For last 10: 
        # If accuracy on last 10 is greater than 60%, target is presented 20ms shorter.
        # If accuracy on last 10 is less than 60%, target is presented 50ms longer.
        # If accuracy on last 10 is 60%, target presentation time does not change. 
        if trial_num <= 4:
            resp_time = resp_time;
            thisExp.addData('full_target_time', resp_time);
        elif trial_num > 5 and trial_num <= 8:
            last_6_acc = calibration_accuracy[-6:];
            acc_ratio = sum(last_6_acc) / len(last_6_acc);
            thisExp.addData('acc_ratio', acc_ratio);
            if acc_ratio <= 0.66: #target presented for 50 ms longer
                resp_time = resp_time + 0.05
                thisExp.addData('full_target_time', resp_time);
            else: #target presented for 20ms shorter
                resp_time = resp_time - 0.02
                thisExp.addData('full_target_time', resp_time);
        else:
            last_10_acc = calibration_accuracy[-10:];
            acc_ratio = sum(last_10_acc) / len(last_10_acc);
            thisExp.addData('acc_ratio', acc_ratio);
            if acc_ratio < 0.60: #target presented for 50 ms longer
                resp_time = resp_time + 0.05
                thisExp.addData('full_target_time', resp_time);
            elif acc_ratio == 0.60: #target presentation time stays the same. 
                resp_time = resp_time
                thisExp.addData('full_target_time', resp_time);
            else: #target presented for 20ms shorter
                resp_time = resp_time - 0.02
                thisExp.addData('full_target_time', resp_time);
        ButtonPressTarget.keys = []
        ButtonPressTarget.rt = []
        _ButtonPressTarget_allKeys = []
        # keep track of which components have finished
        TargetPresentationComponents = [TargetPresentationScreen, ButtonPressTarget]
        for thisComponent in TargetPresentationComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "TargetPresentation" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *TargetPresentationScreen* updates
            
            # if TargetPresentationScreen is starting this frame...
            if TargetPresentationScreen.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                
                # Send LSL Marker :
                mark = 'Target'
                outlet.push_sample([mark])  # Push event marker.                   
                
                # keep track of start time/frame for later
                TargetPresentationScreen.frameNStart = frameN  # exact frame index
                TargetPresentationScreen.tStart = t  # local t and not account for scr refresh
                TargetPresentationScreen.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(TargetPresentationScreen, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'TargetPresentationScreen.started')
                # update status
                TargetPresentationScreen.status = STARTED
                TargetPresentationScreen.setAutoDraw(True)
            
            # if TargetPresentationScreen is active this frame...
            if TargetPresentationScreen.status == STARTED:
                # update params
                pass
            
            # if TargetPresentationScreen is stopping this frame...
            if TargetPresentationScreen.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > TargetPresentationScreen.tStartRefresh + resp_time-frameTolerance:
                    # keep track of stop time/frame for later
                    TargetPresentationScreen.tStop = t  # not accounting for scr refresh
                    TargetPresentationScreen.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'TargetPresentationScreen.stopped')
                    # update status
                    TargetPresentationScreen.status = FINISHED
                    TargetPresentationScreen.setAutoDraw(False)
            
            # *ButtonPressTarget* updates
            waitOnFlip = False
            
            # if ButtonPressTarget is starting this frame...
            if ButtonPressTarget.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                ButtonPressTarget.frameNStart = frameN  # exact frame index
                ButtonPressTarget.tStart = t  # local t and not account for scr refresh
                ButtonPressTarget.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(ButtonPressTarget, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'ButtonPressTarget.started')
                # update status
                ButtonPressTarget.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(ButtonPressTarget.clock.reset)  # t=0 on next screen flip
            
            # if ButtonPressTarget is stopping this frame...
            if ButtonPressTarget.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > ButtonPressTarget.tStartRefresh + resp_time-frameTolerance:
                    # keep track of stop time/frame for later
                    ButtonPressTarget.tStop = t  # not accounting for scr refresh
                    ButtonPressTarget.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'ButtonPressTarget.stopped')
                    # update status
                    ButtonPressTarget.status = FINISHED
                    ButtonPressTarget.status = FINISHED
            if ButtonPressTarget.status == STARTED and not waitOnFlip:
                theseKeys = ButtonPressTarget.getKeys(keyList=['left','right'], ignoreKeys=["escape"], waitRelease=False)
                _ButtonPressTarget_allKeys.extend(theseKeys)
                if len(_ButtonPressTarget_allKeys):
                    ButtonPressTarget.keys = _ButtonPressTarget_allKeys[0].name  # just the first key pressed
                    ButtonPressTarget.rt = _ButtonPressTarget_allKeys[0].rt
                    ButtonPressTarget.duration = _ButtonPressTarget_allKeys[0].duration
                    # was this correct?
                    if (ButtonPressTarget.keys == str(corr_button)) or (ButtonPressTarget.keys == corr_button):
                        ButtonPressTarget.corr = 1
                    else:
                        ButtonPressTarget.corr = 0
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in TargetPresentationComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "TargetPresentation" ---
        for thisComponent in TargetPresentationComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('TargetPresentation.stopped', globalClock.getTime())
        # check responses
        if ButtonPressTarget.keys in ['', [], None]:  # No response was made
            ButtonPressTarget.keys = None
            # was no response the correct answer?!
            if str(corr_button).lower() == 'none':
               ButtonPressTarget.corr = 1;  # correct non-response
            else:
               ButtonPressTarget.corr = 0;  # failed to respond (incorrectly)
        # store data for MainLoop (TrialHandler)
        MainLoop.addData('ButtonPressTarget.keys',ButtonPressTarget.keys)
        MainLoop.addData('ButtonPressTarget.corr', ButtonPressTarget.corr)

        if ButtonPressTarget.keys == None and EarlyPressFixation.keys == None and EarlyPressCue.keys == None:
            resp = "Late"
            print("Response: [%s]" % resp)
            outlet.push_sample([resp])  # Push event marker
        if ButtonPressTarget.keys != None and EarlyPressFixation.keys == None and EarlyPressCue.keys == None:  # we had a response only at target routine
            # Send LSL Marker : Response as string
            if ButtonPressTarget.keys == corr_button:
                mystring = ' '.join(map(str,ButtonPressTarget.keys))
                resp = "Correct"
                print("Response: [%s]" % resp)    
                outlet.push_sample([resp])  # Push event marker
            else:
                mystring = ' '.join(map(str,ButtonPressTarget.keys))
                resp = "Incorrect"
                print("Response: [%s]" % resp)    
                outlet.push_sample([resp])  # Push event marker

        if ButtonPressTarget.keys != None and EarlyPressFixation.keys == None and EarlyPressCue.keys == None:  # we had a response only at target routine
            MainLoop.addData('ButtonPressTarget.rt', ButtonPressTarget.rt)
            MainLoop.addData('ButtonPressTarget.duration', ButtonPressTarget.duration)
        # Run 'End Routine' code from codeFeedbacksaving
        feedbackver = [];
        # Used to decide feedback participants get based on cue (win, loss, neutral) and button press
        if EarlyPressCue.keys != None or EarlyPressFixation.keys != None and colour == "blue":
            feedbackver = "3"; # Early press win cue -> did not win point.
            thisExp.addData('outcome_label', "Early")
            #thisExp.addData('outcome_val', -2)
            calibration_accuracy.append(0)
        elif EarlyPressCue.keys != None or EarlyPressFixation.keys != None and colour == "red":
            feedbackver = "4"; # Early press loss cue -> lost point.
            thisExp.addData('outcome_label', "Early")
            #thisExp.addData('outcome_val', -2)
            calibration_accuracy.append(0)
        elif EarlyPressCue.keys != None or EarlyPressFixation.keys != None and colour == "yellow":
            feedbackver = "6"; # Early press neutral cue -> points stay the same, but incorrect.
            thisExp.addData('outcome_label', "Early")
            #thisExp.addData('outcome_val', -2)
            calibration_accuracy.append(0)
        elif ButtonPressTarget.corr == 1 and colour == "blue":
            feedbackver = "1"; # Correct press win cue -> won point.
            thisExp.addData('outcome_label', "Correct")
            #thisExp.addData('outcome_val', 1)
            calibration_accuracy.append(1)
        elif ButtonPressTarget.corr == 1 and colour == "red":
            feedbackver = "2"; # Correct press loss cue -> did not lose point.
            thisExp.addData('outcome_label', "Correct")
            #thisExp.addData('outcome_val', 1)
            calibration_accuracy.append(1)
        elif ButtonPressTarget.corr == 1 and colour == "yellow":
            feedbackver = "5"; # Correct press neutral cue -> points stay the same.
            thisExp.addData('outcome_label', "Correct")
            #thisExp.addData('outcome_val', 1)
            calibration_accuracy.append(1)
        elif ButtonPressTarget.corr == 0 and colour == "blue":
            feedbackver = "3"; # Incorrect press win cue -> did not win point.
            thisExp.addData('outcome_label', "Incorrect")
            #thisExp.addData('outcome_val', -1)
            calibration_accuracy.append(0)
        elif ButtonPressTarget.corr == 0 and colour == "yellow":
            feedbackver = "6"; # Incorrect press neutral cue -> points stay the same.
            thisExp.addData('outcome_label', "Incorrect")
            #thisExp.addData('outcome_val', -1)
            calibration_accuracy.append(0)
        else:
            feedbackver = "4"; # Incorrect press loss cue -> lost point.
            thisExp.addData('outcome_label', "Incorrect")
            #thisExp.addData('outcome_val', -1)
            calibration_accuracy.append(0)
            
        # the Routine "TargetPresentation" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "FeedbackCode" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('FeedbackCode.started', globalClock.getTime())
        # Run 'Begin Routine' code from feedbacktextcode
        # Correct press win cue (blue)
        if feedbackver == "1":
            text = "Sie haben einen Punkt gewonnen";
            textcolour = 'green';
            treat_counter += 1;
            
        # Correct press loss cue (red)
        elif feedbackver == "2":
            text = "Sie haben keinen Punkt verloren";
            textcolour = 'green';
            
        # Incorrect press win cue (blue)    
        elif feedbackver == "3":
            text = "Sie haben keinen Punkt gewonnen";
            textcolour = 'red';
            
        # Incorrect press loss cue (red)        
        elif feedbackver == "4":
            text = "Sie haben einen Punkt verloren";
            textcolour = 'red';
            treat_counter -= 1;
            
        # Neutral cue, correct press, points unchanged (yellow)
        elif feedbackver == "5":
            text = "Sie haben keinen Punkt verloren";
            textcolour = 'green';
            
        # Neutral cue, incorrect press, points unchanged (yellow)
        else: #Feedback version 6
            text = "Sie haben keinen Punkt verloren";
            textcolour = 'red';
        
        text_Feedback.setColor(textcolour, colorSpace='rgb')
        text_Feedback.setText(text)
        text_treatCounter.setText(treat_counter)
        # keep track of which components have finished
        FeedbackCodeComponents = [text_Feedback, text_treatCounter]
        for thisComponent in FeedbackCodeComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "FeedbackCode" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # is it time to end the Routine? (based on local clock)
            if tThisFlip > 1-frameTolerance:
                continueRoutine = False
            
            # *text_Feedback* updates
            
            # if text_Feedback is starting this frame...
            if text_Feedback.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                
                # Send LSL Marker : 
                mark = 'Feedback'
                outlet.push_sample([mark])  # Push event marker.                
                
                # keep track of start time/frame for later
                text_Feedback.frameNStart = frameN  # exact frame index
                text_Feedback.tStart = t  # local t and not account for scr refresh
                text_Feedback.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_Feedback, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_Feedback.started')
                # update status
                text_Feedback.status = STARTED
                text_Feedback.setAutoDraw(True)
            
            # if text_Feedback is active this frame...
            if text_Feedback.status == STARTED:
                # update params
                pass
            
            # if text_Feedback is stopping this frame...
            if text_Feedback.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_Feedback.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    text_Feedback.tStop = t  # not accounting for scr refresh
                    text_Feedback.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_Feedback.stopped')
                    # update status
                    text_Feedback.status = FINISHED
                    text_Feedback.setAutoDraw(False)
            
            # *text_treatCounter* updates
            
            # if text_treatCounter is starting this frame...
            if text_treatCounter.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                text_treatCounter.frameNStart = frameN  # exact frame index
                text_treatCounter.tStart = t  # local t and not account for scr refresh
                text_treatCounter.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_treatCounter, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_treatCounter.started')
                # update status
                text_treatCounter.status = STARTED
                text_treatCounter.setAutoDraw(True)
            
            # if text_treatCounter is active this frame...
            if text_treatCounter.status == STARTED:
                # update params
                pass
            
            # if text_treatCounter is stopping this frame...
            if text_treatCounter.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_treatCounter.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    text_treatCounter.tStop = t  # not accounting for scr refresh
                    text_treatCounter.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_treatCounter.stopped')
                    # update status
                    text_treatCounter.status = FINISHED
                    text_treatCounter.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in FeedbackCodeComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "FeedbackCode" ---
        for thisComponent in FeedbackCodeComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('FeedbackCode.stopped', globalClock.getTime())
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        
        # --- Prepare to start Routine "ITI500" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('ITI500.started', globalClock.getTime())
        # keep track of which components have finished
        ITI500Components = [trial_ITI]
        for thisComponent in ITI500Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "ITI500" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.5:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *trial_ITI* updates
            
            # if trial_ITI is starting this frame...
            if trial_ITI.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                
                # Send LSL Marker : 
                mark = 'ITI'
                outlet.push_sample([mark])  # Push event marker.                
                
                # keep track of start time/frame for later
                trial_ITI.frameNStart = frameN  # exact frame index
                trial_ITI.tStart = t  # local t and not account for scr refresh
                trial_ITI.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(trial_ITI, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'trial_ITI.started')
                # update status
                trial_ITI.status = STARTED
                trial_ITI.setAutoDraw(True)
            
            # if trial_ITI is active this frame...
            if trial_ITI.status == STARTED:
                # update params
                pass
            
            # if trial_ITI is stopping this frame...
            if trial_ITI.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > trial_ITI.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    trial_ITI.tStop = t  # not accounting for scr refresh
                    trial_ITI.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'trial_ITI.stopped')
                    # update status
                    trial_ITI.status = FINISHED
                    trial_ITI.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in ITI500Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "ITI500" ---
        for thisComponent in ITI500Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('ITI500.stopped', globalClock.getTime())
        # Run 'End Routine' code from trial_num_code
        # Used in codeTargetPresTiming
        trial_num = trial_num + 1;
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.500000)
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 15.0 repeats of 'MainLoop'
    
    
    # --- Prepare to start Routine "EndScreen" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('EndScreen.started', globalClock.getTime())
    # keep track of which components have finished
    EndScreenComponents = [EndScreenText]
    for thisComponent in EndScreenComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "EndScreen" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 3.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *EndScreenText* updates
        
        # if EndScreenText is starting this frame...
        if EndScreenText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            EndScreenText.frameNStart = frameN  # exact frame index
            EndScreenText.tStart = t  # local t and not account for scr refresh
            EndScreenText.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(EndScreenText, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'EndScreenText.started')
            # update status
            EndScreenText.status = STARTED
            EndScreenText.setAutoDraw(True)
        
        # if EndScreenText is active this frame...
        if EndScreenText.status == STARTED:
            # update params
            pass
        
        # if EndScreenText is stopping this frame...
        if EndScreenText.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > EndScreenText.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                EndScreenText.tStop = t  # not accounting for scr refresh
                EndScreenText.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'EndScreenText.stopped')
                # update status
                EndScreenText.status = FINISHED
                EndScreenText.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in EndScreenComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "EndScreen" ---
    for thisComponent in EndScreenComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('EndScreen.stopped', globalClock.getTime())
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-3.000000)
    
    # mark experiment as finished
    endExperiment(thisExp, win=win, inputs=inputs)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, inputs=None, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    inputs : dict
        Dictionary of input devices by name.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # shut down eyetracker, if there is one
    if inputs is not None:
        if 'eyetracker' in inputs and inputs['eyetracker'] is not None:
            inputs['eyetracker'].setConnectionState(False)
    logging.flush()


def quit(thisExp, win=None, inputs=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    inputs : dict
        Dictionary of input devices by name.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    if inputs is not None:
        if 'eyetracker' in inputs and inputs['eyetracker'] is not None:
            inputs['eyetracker'].setConnectionState(False)
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    inputs = setupInputs(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win, 
        inputs=inputs
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win, inputs=inputs)
