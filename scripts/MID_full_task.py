#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2023.2.3),
    on Fri Mar  1 09:15:40 2024
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
        originPath='/Users/charlotte/Dropbox/Charite_PhD/tasks/MID_py/LSL_full_MID.py',
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
            size=[800, 480], fullscr=False, screen=0,
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
    win.mouseVisible = True
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
    
    # --- Initialize components for Routine "practice_WelcomeScreen" ---
    practice_StartScreen_ButtonPress = keyboard.Keyboard()
    practice_textStartScreen = visual.TextStim(win=win, name='practice_textStartScreen',
        text='Übungssitzungen!\nDrücken Sie zum Starten die rechte Taste',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    # Run 'Begin Experiment' code from LSL_initiation_practice_ITI_code
    # Initiate LSL
    import pylsl
    
    # Make stream outlets & info for each "marker" we want to push, and a corresponding outlet
    print('Creating a new streaminfo...')
    screen_info = pylsl.StreamInfo('screen_markers', 'screen_pres', 1, 0, 'string')
    behav_info = pylsl.StreamInfo('button_press', 'beh', 1, 0, 'string')
    
    print('Opening an outlet...')
    screen_outlet = pylsl.StreamOutlet(screen_info)
    behav_outlet = pylsl.StreamOutlet(behav_info)
    
    print("now sending markers...")
    screen_markers = ['Cue_win', 'Cue_loss', 'Fixation', 'Target', 'Feedback', 'ITI']
    behav_markers = ['Early', 'Correct', 'Incorrect']
    
    # --- Initialize components for Routine "practice_first_ITI" ---
    practice_first_iti_fig = visual.ShapeStim(
        win=win, name='practice_first_iti_fig', vertices='cross',
        size=(0.01, 0.01),
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    # Run 'Begin Experiment' code from practice_ITI_code
    # Variables used for calculating target presentation time 
    # and score count in practice loop
    practice_treat_counter = 0;
    practice_target_pres_time = 0.4;
    practice_trial_num = 0;
    practice_calibration_accuracy = [];
    
    # --- Initialize components for Routine "practice_CuePresentation" ---
    practice_CueCircle = visual.ShapeStim(
        win=win, name='practice_CueCircle',
        size=(0.25, 0.25), vertices='circle',
        ori=0.0, pos=[0,0], anchor='center-left',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    practice_EarlyPressCue = keyboard.Keyboard()
    
    # --- Initialize components for Routine "practice_Fixation2" ---
    practice_FixationScreen = visual.ShapeStim(
        win=win, name='practice_FixationScreen', vertices='cross',
        size=(0.01, 0.01),
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    practice_EarlyPressFixation = keyboard.Keyboard()
    
    # --- Initialize components for Routine "practice_TargetPresentation" ---
    practice_TargetPresentationScreen = visual.ShapeStim(
        win=win, name='practice_TargetPresentationScreen',
        size=(0.25, 0.25), vertices='circle',
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-1.0, interpolate=True)
    practice_ButtonPressTarget = keyboard.Keyboard()
    
    # --- Initialize components for Routine "practice_FeedbackCode" ---
    practice_text_Feedback = visual.TextStim(win=win, name='practice_text_Feedback',
        text='',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    practice_text_treatCounter = visual.TextStim(win=win, name='practice_text_treatCounter',
        text='',
        font='Open Sans',
        pos=(0, -0.25), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "practice_ITI500" ---
    practice_trial_ITI = visual.ShapeStim(
        win=win, name='practice_trial_ITI', vertices='cross',
        size=(0.01, 0.01),
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "practice_EndScreen" ---
    practice_EndScreenText = visual.TextStim(win=win, name='practice_EndScreenText',
        text='Ende der Übungssitzungen.',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "WelcomeScreen" ---
    StartScreen_ButtonPress = keyboard.Keyboard()
    textStartScreen = visual.TextStim(win=win, name='textStartScreen',
        text='Drücken Sie zum Starten die rechte Taste',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "first_ITI" ---
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
    target_pres_time = 0.4;
    trial_num = 0;
    calibration_accuracy = [];
    
    # --- Initialize components for Routine "CuePresentation" ---
    CueCircle = visual.ShapeStim(
        win=win, name='CueCircle',
        size=(0.25, 0.25), vertices='circle',
        ori=0.0, pos=[0,0], anchor='center-left',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    EarlyPressCue = keyboard.Keyboard()
    
    # --- Initialize components for Routine "Fixation2" ---
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
        text='Danke für die Teilnahme!',
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
    
    # --- Prepare to start Routine "practice_WelcomeScreen" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('practice_WelcomeScreen.started', globalClock.getTime())
    practice_StartScreen_ButtonPress.keys = []
    practice_StartScreen_ButtonPress.rt = []
    _practice_StartScreen_ButtonPress_allKeys = []
    # keep track of which components have finished
    practice_WelcomeScreenComponents = [practice_StartScreen_ButtonPress, practice_textStartScreen]
    for thisComponent in practice_WelcomeScreenComponents:
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
    
    # --- Run Routine "practice_WelcomeScreen" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *practice_StartScreen_ButtonPress* updates
        waitOnFlip = False
        
        # if practice_StartScreen_ButtonPress is starting this frame...
        if practice_StartScreen_ButtonPress.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            practice_StartScreen_ButtonPress.frameNStart = frameN  # exact frame index
            practice_StartScreen_ButtonPress.tStart = t  # local t and not account for scr refresh
            practice_StartScreen_ButtonPress.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(practice_StartScreen_ButtonPress, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'practice_StartScreen_ButtonPress.started')
            # update status
            practice_StartScreen_ButtonPress.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(practice_StartScreen_ButtonPress.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(practice_StartScreen_ButtonPress.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if practice_StartScreen_ButtonPress.status == STARTED and not waitOnFlip:
            theseKeys = practice_StartScreen_ButtonPress.getKeys(keyList=['right'], ignoreKeys=["escape"], waitRelease=False)
            _practice_StartScreen_ButtonPress_allKeys.extend(theseKeys)
            if len(_practice_StartScreen_ButtonPress_allKeys):
                practice_StartScreen_ButtonPress.keys = _practice_StartScreen_ButtonPress_allKeys[-1].name  # just the last key pressed
                practice_StartScreen_ButtonPress.rt = _practice_StartScreen_ButtonPress_allKeys[-1].rt
                practice_StartScreen_ButtonPress.duration = _practice_StartScreen_ButtonPress_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *practice_textStartScreen* updates
        
        # if practice_textStartScreen is starting this frame...
        if practice_textStartScreen.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            practice_textStartScreen.frameNStart = frameN  # exact frame index
            practice_textStartScreen.tStart = t  # local t and not account for scr refresh
            practice_textStartScreen.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(practice_textStartScreen, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'practice_textStartScreen.started')
            # update status
            practice_textStartScreen.status = STARTED
            practice_textStartScreen.setAutoDraw(True)
        
        # if practice_textStartScreen is active this frame...
        if practice_textStartScreen.status == STARTED:
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
        for thisComponent in practice_WelcomeScreenComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "practice_WelcomeScreen" ---
    for thisComponent in practice_WelcomeScreenComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('practice_WelcomeScreen.stopped', globalClock.getTime())
    # check responses
    if practice_StartScreen_ButtonPress.keys in ['', [], None]:  # No response was made
        practice_StartScreen_ButtonPress.keys = None
    thisExp.addData('practice_StartScreen_ButtonPress.keys',practice_StartScreen_ButtonPress.keys)
    if practice_StartScreen_ButtonPress.keys != None:  # we had a response
        thisExp.addData('practice_StartScreen_ButtonPress.rt', practice_StartScreen_ButtonPress.rt)
        thisExp.addData('practice_StartScreen_ButtonPress.duration', practice_StartScreen_ButtonPress.duration)
    thisExp.nextEntry()
    # the Routine "practice_WelcomeScreen" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "practice_first_ITI" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('practice_first_ITI.started', globalClock.getTime())
    # keep track of which components have finished
    practice_first_ITIComponents = [practice_first_iti_fig]
    for thisComponent in practice_first_ITIComponents:
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
    
    # --- Run Routine "practice_first_ITI" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 0.5:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *practice_first_iti_fig* updates
        
        # if practice_first_iti_fig is starting this frame...
        if practice_first_iti_fig.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            practice_first_iti_fig.frameNStart = frameN  # exact frame index
            practice_first_iti_fig.tStart = t  # local t and not account for scr refresh
            practice_first_iti_fig.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(practice_first_iti_fig, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'practice_first_iti_fig.started')
            # update status
            practice_first_iti_fig.status = STARTED
            practice_first_iti_fig.setAutoDraw(True)
        
        # if practice_first_iti_fig is active this frame...
        if practice_first_iti_fig.status == STARTED:
            # update params
            pass
        
        # if practice_first_iti_fig is stopping this frame...
        if practice_first_iti_fig.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > practice_first_iti_fig.tStartRefresh + 0.5-frameTolerance:
                # keep track of stop time/frame for later
                practice_first_iti_fig.tStop = t  # not accounting for scr refresh
                practice_first_iti_fig.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'practice_first_iti_fig.stopped')
                # update status
                practice_first_iti_fig.status = FINISHED
                practice_first_iti_fig.setAutoDraw(False)
        
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
        for thisComponent in practice_first_ITIComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "practice_first_ITI" ---
    for thisComponent in practice_first_ITIComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('practice_first_ITI.stopped', globalClock.getTime())
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-0.500000)
    
    # set up handler to look after randomisation of conditions etc
    PracticeLoop = data.TrialHandler(nReps=6.0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('practiceloop_conditions.xlsx'),
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
        
        # --- Prepare to start Routine "practice_CuePresentation" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('practice_CuePresentation.started', globalClock.getTime())
        practice_CueCircle.setFillColor(practice_colour)
        practice_CueCircle.setPos(practice_location)
        practice_CueCircle.setLineColor(practice_colour)
        practice_EarlyPressCue.keys = []
        practice_EarlyPressCue.rt = []
        _practice_EarlyPressCue_allKeys = []
        # Run 'Begin Routine' code from practice_CuePres_LSLcode
        # #Pushes screen marker
        #if practice_colour == 'blue': 
        #    screen_outlet.push_sample([screen_markers[0]]) #pushes cue_win marker
        #elif practice_colour == 'yellow':
        #    screen_outlet.push_sample([screen_markers[1]]) #pushes cue_loss marker
        
        # #counter used to push one marker per trial, regardless of number of button presses.
        #practice_CuePres_marker_count = 0 
        
        # keep track of which components have finished
        practice_CuePresentationComponents = [practice_CueCircle, practice_EarlyPressCue]
        for thisComponent in practice_CuePresentationComponents:
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
        
        # --- Run Routine "practice_CuePresentation" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.25:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *practice_CueCircle* updates
            
            # if practice_CueCircle is starting this frame...
            if practice_CueCircle.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                practice_CueCircle.frameNStart = frameN  # exact frame index
                practice_CueCircle.tStart = t  # local t and not account for scr refresh
                practice_CueCircle.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(practice_CueCircle, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'practice_CueCircle.started')
                # update status
                practice_CueCircle.status = STARTED
                practice_CueCircle.setAutoDraw(True)
            
            # if practice_CueCircle is active this frame...
            if practice_CueCircle.status == STARTED:
                # update params
                pass
            
            # if practice_CueCircle is stopping this frame...
            if practice_CueCircle.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > practice_CueCircle.tStartRefresh + 0.25-frameTolerance:
                    # keep track of stop time/frame for later
                    practice_CueCircle.tStop = t  # not accounting for scr refresh
                    practice_CueCircle.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'practice_CueCircle.stopped')
                    # update status
                    practice_CueCircle.status = FINISHED
                    practice_CueCircle.setAutoDraw(False)
            
            # *practice_EarlyPressCue* updates
            waitOnFlip = False
            
            # if practice_EarlyPressCue is starting this frame...
            if practice_EarlyPressCue.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                practice_EarlyPressCue.frameNStart = frameN  # exact frame index
                practice_EarlyPressCue.tStart = t  # local t and not account for scr refresh
                practice_EarlyPressCue.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(practice_EarlyPressCue, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'practice_EarlyPressCue.started')
                # update status
                practice_EarlyPressCue.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(practice_EarlyPressCue.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(practice_EarlyPressCue.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if practice_EarlyPressCue is stopping this frame...
            if practice_EarlyPressCue.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > practice_EarlyPressCue.tStartRefresh + 0.25-frameTolerance:
                    # keep track of stop time/frame for later
                    practice_EarlyPressCue.tStop = t  # not accounting for scr refresh
                    practice_EarlyPressCue.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'practice_EarlyPressCue.stopped')
                    # update status
                    practice_EarlyPressCue.status = FINISHED
                    practice_EarlyPressCue.status = FINISHED
            if practice_EarlyPressCue.status == STARTED and not waitOnFlip:
                theseKeys = practice_EarlyPressCue.getKeys(keyList=['left', 'right'], ignoreKeys=["escape"], waitRelease=False)
                _practice_EarlyPressCue_allKeys.extend(theseKeys)
                if len(_practice_EarlyPressCue_allKeys):
                    practice_EarlyPressCue.keys = _practice_EarlyPressCue_allKeys[0].name  # just the first key pressed
                    practice_EarlyPressCue.rt = _practice_EarlyPressCue_allKeys[0].rt
                    practice_EarlyPressCue.duration = _practice_EarlyPressCue_allKeys[0].duration
            # Run 'Each Frame' code from practice_CuePres_LSLcode
            # #'Early' button marker sent if they press
            #if 'right' in practice_EarlyPressCue.keys or 'left' in practice_EarlyPressCue.keys:
            #    if practice_CuePres_marker_count == 0:
            #        behav_outlet.push_sample([behav_markers[0]])
            #        practice_CuePres_marker_count += 1
            
                   
                
            
            
            
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
            for thisComponent in practice_CuePresentationComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "practice_CuePresentation" ---
        for thisComponent in practice_CuePresentationComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('practice_CuePresentation.stopped', globalClock.getTime())
        # check responses
        if practice_EarlyPressCue.keys in ['', [], None]:  # No response was made
            practice_EarlyPressCue.keys = None
        PracticeLoop.addData('practice_EarlyPressCue.keys',practice_EarlyPressCue.keys)
        if practice_EarlyPressCue.keys != None:  # we had a response
            PracticeLoop.addData('practice_EarlyPressCue.rt', practice_EarlyPressCue.rt)
            PracticeLoop.addData('practice_EarlyPressCue.duration', practice_EarlyPressCue.duration)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.250000)
        
        # --- Prepare to start Routine "practice_Fixation2" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('practice_Fixation2.started', globalClock.getTime())
        practice_EarlyPressFixation.keys = []
        practice_EarlyPressFixation.rt = []
        _practice_EarlyPressFixation_allKeys = []
        # Run 'Begin Routine' code from practice_fixation_LSL_code
        #screen_outlet.push_sample([screen_markers[2]]) #pushes fixation marker
        #practice_fixation_marker_count = 0 #used in fixation button presses
        
        # keep track of which components have finished
        practice_Fixation2Components = [practice_FixationScreen, practice_EarlyPressFixation]
        for thisComponent in practice_Fixation2Components:
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
        
        # --- Run Routine "practice_Fixation2" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *practice_FixationScreen* updates
            
            # if practice_FixationScreen is starting this frame...
            if practice_FixationScreen.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                practice_FixationScreen.frameNStart = frameN  # exact frame index
                practice_FixationScreen.tStart = t  # local t and not account for scr refresh
                practice_FixationScreen.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(practice_FixationScreen, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'practice_FixationScreen.started')
                # update status
                practice_FixationScreen.status = STARTED
                practice_FixationScreen.setAutoDraw(True)
            
            # if practice_FixationScreen is active this frame...
            if practice_FixationScreen.status == STARTED:
                # update params
                pass
            
            # if practice_FixationScreen is stopping this frame...
            if practice_FixationScreen.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > practice_FixationScreen.tStartRefresh + 2-frameTolerance:
                    # keep track of stop time/frame for later
                    practice_FixationScreen.tStop = t  # not accounting for scr refresh
                    practice_FixationScreen.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'practice_FixationScreen.stopped')
                    # update status
                    practice_FixationScreen.status = FINISHED
                    practice_FixationScreen.setAutoDraw(False)
            
            # *practice_EarlyPressFixation* updates
            waitOnFlip = False
            
            # if practice_EarlyPressFixation is starting this frame...
            if practice_EarlyPressFixation.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                practice_EarlyPressFixation.frameNStart = frameN  # exact frame index
                practice_EarlyPressFixation.tStart = t  # local t and not account for scr refresh
                practice_EarlyPressFixation.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(practice_EarlyPressFixation, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'practice_EarlyPressFixation.started')
                # update status
                practice_EarlyPressFixation.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(practice_EarlyPressFixation.clock.reset)  # t=0 on next screen flip
            
            # if practice_EarlyPressFixation is stopping this frame...
            if practice_EarlyPressFixation.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > practice_EarlyPressFixation.tStartRefresh + 2.0-frameTolerance:
                    # keep track of stop time/frame for later
                    practice_EarlyPressFixation.tStop = t  # not accounting for scr refresh
                    practice_EarlyPressFixation.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'practice_EarlyPressFixation.stopped')
                    # update status
                    practice_EarlyPressFixation.status = FINISHED
                    practice_EarlyPressFixation.status = FINISHED
            if practice_EarlyPressFixation.status == STARTED and not waitOnFlip:
                theseKeys = practice_EarlyPressFixation.getKeys(keyList=['left','right'], ignoreKeys=["escape"], waitRelease=False)
                _practice_EarlyPressFixation_allKeys.extend(theseKeys)
                if len(_practice_EarlyPressFixation_allKeys):
                    practice_EarlyPressFixation.keys = _practice_EarlyPressFixation_allKeys[0].name  # just the first key pressed
                    practice_EarlyPressFixation.rt = _practice_EarlyPressFixation_allKeys[0].rt
                    practice_EarlyPressFixation.duration = _practice_EarlyPressFixation_allKeys[0].duration
            # Run 'Each Frame' code from practice_fixation_LSL_code
            # #pushes button marker if they press early
            #if 'right' in practice_EarlyPressFixation.keys or 'left' in practice_EarlyPressFixation.keys:
            #    if practice_fixation_marker_count == 0:
            #        behav_outlet.push_sample([behav_markers[0]])
            #        practice_fixation_marker_count += 1 #this avoids multiple markers to be sent in same trial
            
                   
                
            
            
            
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
            for thisComponent in practice_Fixation2Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "practice_Fixation2" ---
        for thisComponent in practice_Fixation2Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('practice_Fixation2.stopped', globalClock.getTime())
        # check responses
        if practice_EarlyPressFixation.keys in ['', [], None]:  # No response was made
            practice_EarlyPressFixation.keys = None
        PracticeLoop.addData('practice_EarlyPressFixation.keys',practice_EarlyPressFixation.keys)
        if practice_EarlyPressFixation.keys != None:  # we had a response
            PracticeLoop.addData('practice_EarlyPressFixation.rt', practice_EarlyPressFixation.rt)
            PracticeLoop.addData('practice_EarlyPressFixation.duration', practice_EarlyPressFixation.duration)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.000000)
        
        # --- Prepare to start Routine "practice_TargetPresentation" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('practice_TargetPresentation.started', globalClock.getTime())
        # Run 'Begin Routine' code from practice_codeTargetPresTiming
        # Calculates target presentation time, based on performance of last 6 trials.
        # If accuracy on last 6 is greater than 66%, target is presented 20ms shorter.
        # If accuract on last 6 is less than 66%, target is presented 50ms longer.
        if practice_trial_num <= 5:
            practice_target_pres_time = practice_target_pres_time;
            thisExp.addData('practice_target_pres_time', practice_target_pres_time);
        else:
            practice_last_6_acc = practice_calibration_accuracy[practice_trial_num-5:practice_trial_num+1];
            practice_acc_ratio = sum(practice_last_6_acc) / len(practice_last_6_acc);
            if practice_acc_ratio <= 0.66: #target presented for 50 ms longer
                practice_target_pres_time = practice_target_pres_time + 0.05
                thisExp.addData('practice_target_pres_time', practice_target_pres_time);
            else: #target presented for 20ms shorter
                practice_target_pres_time = practice_target_pres_time - 0.02
                thisExp.addData('practice_target_pres_time', practice_target_pres_time);
        practice_ButtonPressTarget.keys = []
        practice_ButtonPressTarget.rt = []
        _practice_ButtonPressTarget_allKeys = []
        # Run 'Begin Routine' code from practice_TargetPres_LSL_code
        #screen_outlet.push_sample([screen_markers[3]]) #pushes targetpres marker
        #practice_TargetPres_marker_count = 0 # used in button press marker, avoids multiple markers being pushed
        # keep track of which components have finished
        practice_TargetPresentationComponents = [practice_TargetPresentationScreen, practice_ButtonPressTarget]
        for thisComponent in practice_TargetPresentationComponents:
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
        
        # --- Run Routine "practice_TargetPresentation" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *practice_TargetPresentationScreen* updates
            
            # if practice_TargetPresentationScreen is starting this frame...
            if practice_TargetPresentationScreen.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                practice_TargetPresentationScreen.frameNStart = frameN  # exact frame index
                practice_TargetPresentationScreen.tStart = t  # local t and not account for scr refresh
                practice_TargetPresentationScreen.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(practice_TargetPresentationScreen, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'practice_TargetPresentationScreen.started')
                # update status
                practice_TargetPresentationScreen.status = STARTED
                practice_TargetPresentationScreen.setAutoDraw(True)
            
            # if practice_TargetPresentationScreen is active this frame...
            if practice_TargetPresentationScreen.status == STARTED:
                # update params
                pass
            
            # if practice_TargetPresentationScreen is stopping this frame...
            if practice_TargetPresentationScreen.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > practice_TargetPresentationScreen.tStartRefresh + target_pres_time-frameTolerance:
                    # keep track of stop time/frame for later
                    practice_TargetPresentationScreen.tStop = t  # not accounting for scr refresh
                    practice_TargetPresentationScreen.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'practice_TargetPresentationScreen.stopped')
                    # update status
                    practice_TargetPresentationScreen.status = FINISHED
                    practice_TargetPresentationScreen.setAutoDraw(False)
            
            # *practice_ButtonPressTarget* updates
            waitOnFlip = False
            
            # if practice_ButtonPressTarget is starting this frame...
            if practice_ButtonPressTarget.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                practice_ButtonPressTarget.frameNStart = frameN  # exact frame index
                practice_ButtonPressTarget.tStart = t  # local t and not account for scr refresh
                practice_ButtonPressTarget.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(practice_ButtonPressTarget, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'practice_ButtonPressTarget.started')
                # update status
                practice_ButtonPressTarget.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(practice_ButtonPressTarget.clock.reset)  # t=0 on next screen flip
            
            # if practice_ButtonPressTarget is stopping this frame...
            if practice_ButtonPressTarget.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > practice_ButtonPressTarget.tStartRefresh + target_pres_time-frameTolerance:
                    # keep track of stop time/frame for later
                    practice_ButtonPressTarget.tStop = t  # not accounting for scr refresh
                    practice_ButtonPressTarget.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'practice_ButtonPressTarget.stopped')
                    # update status
                    practice_ButtonPressTarget.status = FINISHED
                    practice_ButtonPressTarget.status = FINISHED
            if practice_ButtonPressTarget.status == STARTED and not waitOnFlip:
                theseKeys = practice_ButtonPressTarget.getKeys(keyList=['left','right'], ignoreKeys=["escape"], waitRelease=False)
                _practice_ButtonPressTarget_allKeys.extend(theseKeys)
                if len(_practice_ButtonPressTarget_allKeys):
                    practice_ButtonPressTarget.keys = _practice_ButtonPressTarget_allKeys[0].name  # just the first key pressed
                    practice_ButtonPressTarget.rt = _practice_ButtonPressTarget_allKeys[0].rt
                    practice_ButtonPressTarget.duration = _practice_ButtonPressTarget_allKeys[0].duration
                    # was this correct?
                    if (practice_ButtonPressTarget.keys == str(practice_corr_button)) or (practice_ButtonPressTarget.keys == practice_corr_button):
                        practice_ButtonPressTarget.corr = 1
                    else:
                        practice_ButtonPressTarget.corr = 0
            # Run 'Each Frame' code from practice_TargetPres_LSL_code
            # Pushes correct or incorrect button marker
            
            #if 'right' in practice_ButtonPressTarget.keys or 'left' in practice_ButtonPressTarget.keys:
            #    if practice_TargetPres_marker_count == 0:
            #        if practice_corr_button == 'left' and practice_ButtonPressTarget.keys == 'left': #correct
            #            behav_outlet.push_sample([behav_markers[1]])
            #            practice_TargetPres_marker_count += 1
            #        elif practice_corr_button == 'right' and practice_ButtonPressTarget.keys == 'right': #correct
            #            behav_outlet.push_sample([behav_markers[1]])
            #            practice_TargetPres_marker_count += 1
            #        else:
            #            behav_outlet.push_sample([behav_markers[2]]) # Incorrect
            #            practice_TargetPres_marker_count += 1
            
            
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
            for thisComponent in practice_TargetPresentationComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "practice_TargetPresentation" ---
        for thisComponent in practice_TargetPresentationComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('practice_TargetPresentation.stopped', globalClock.getTime())
        # check responses
        if practice_ButtonPressTarget.keys in ['', [], None]:  # No response was made
            practice_ButtonPressTarget.keys = None
            # was no response the correct answer?!
            if str(practice_corr_button).lower() == 'none':
               practice_ButtonPressTarget.corr = 1;  # correct non-response
            else:
               practice_ButtonPressTarget.corr = 0;  # failed to respond (incorrectly)
        # store data for PracticeLoop (TrialHandler)
        PracticeLoop.addData('practice_ButtonPressTarget.keys',practice_ButtonPressTarget.keys)
        PracticeLoop.addData('practice_ButtonPressTarget.corr', practice_ButtonPressTarget.corr)
        if practice_ButtonPressTarget.keys != None:  # we had a response
            PracticeLoop.addData('practice_ButtonPressTarget.rt', practice_ButtonPressTarget.rt)
            PracticeLoop.addData('practice_ButtonPressTarget.duration', practice_ButtonPressTarget.duration)
        # Run 'End Routine' code from practice_codeFeedbacksaving
        practice_feedbackver = [];
        # Prepares which feedback version is shown, based on win or loss cues and button press performance. 
        if practice_EarlyPressCue.keys != None or practice_EarlyPressFixation.keys != None and practice_colour == "blue":
            practice_feedbackver = "3"; # Early press win cue -> did not win treat. 
            thisExp.addData('practice_outcome_label', "Early")
            thisExp.addData('practice_outcome_val', -2)
            practice_calibration_accuracy.append(0)
        elif practice_EarlyPressCue.keys != None or practice_EarlyPressFixation.keys != None and practice_colour == "yellow":
            practice_feedbackver = "4"; # Early press loss cue -> lost treat.
            thisExp.addData('practice_outcome_label', "Early")
            thisExp.addData('practice_outcome_val', -2)
            practice_calibration_accuracy.append(0)
        elif practice_ButtonPressTarget.corr == 1 and practice_colour == "blue":
            practice_feedbackver = "1"; # Correct press win cue -> won treat.
            thisExp.addData('practice_outcome_label', "Correct")
            thisExp.addData('practice_outcome_val', 1)
            practice_calibration_accuracy.append(1)
        elif practice_ButtonPressTarget.corr == 1 and practice_colour == "yellow":
            practice_feedbackver = "2"; # Correct press loss cue -> did not lose treat. 
            thisExp.addData('practice_outcome_label', "Correct")
            thisExp.addData('practice_outcome_val', 1)
            practice_calibration_accuracy.append(1)
        elif practice_ButtonPressTarget.corr == 0 and practice_colour == "blue":
            practice_feedbackver = "3"; # Incorrect press win cue -> did not win treat.
            thisExp.addData('practice_outcome_label', "Incorrect")
            thisExp.addData('practice_outcome_val', -1)
            practice_calibration_accuracy.append(0)
        else:
            practice_feedbackver = "4"; # Incorrect press loss cue -> lost treat.
            thisExp.addData('practice_outcome_label', "Incorrect")
            thisExp.addData('practice_outcome_val', -1)
            practice_calibration_accuracy.append(0)
            
        # the Routine "practice_TargetPresentation" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "practice_FeedbackCode" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('practice_FeedbackCode.started', globalClock.getTime())
        # Run 'Begin Routine' code from practice_feedbacktextcode
        if practice_feedbackver == "1":
            practice_text = "Du hast eine Süßigkeit gewonnen";
            practice_textcolour = 'green';
            practice_treat_counter += 1;
            
        elif practice_feedbackver == "2":
            practice_text = "Du hast keine Süßigkeit verloren";
            practice_textcolour = 'green';
            
        elif practice_feedbackver == "3":
            practice_text = "Du hast die Süßigkeiten nicht gewonnen";
            practice_textcolour = 'red';
        
        else:
            practice_text = "Du hast eine Süßigkeit verloren";
            practice_textcolour = 'red';
            practice_treat_counter -= 1;
        
        
        practice_text_Feedback.setColor(practice_textcolour, colorSpace='rgb')
        practice_text_Feedback.setText(practice_text)
        practice_text_treatCounter.setText(practice_treat_counter)
        # Run 'Begin Routine' code from practice_feedback_LSL_code
        #screen_outlet.push_sample([screen_markers[4]]) #pushes fixation marker
        
        # keep track of which components have finished
        practice_FeedbackCodeComponents = [practice_text_Feedback, practice_text_treatCounter]
        for thisComponent in practice_FeedbackCodeComponents:
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
        
        # --- Run Routine "practice_FeedbackCode" ---
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
            
            # *practice_text_Feedback* updates
            
            # if practice_text_Feedback is starting this frame...
            if practice_text_Feedback.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                practice_text_Feedback.frameNStart = frameN  # exact frame index
                practice_text_Feedback.tStart = t  # local t and not account for scr refresh
                practice_text_Feedback.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(practice_text_Feedback, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'practice_text_Feedback.started')
                # update status
                practice_text_Feedback.status = STARTED
                practice_text_Feedback.setAutoDraw(True)
            
            # if practice_text_Feedback is active this frame...
            if practice_text_Feedback.status == STARTED:
                # update params
                pass
            
            # if practice_text_Feedback is stopping this frame...
            if practice_text_Feedback.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > practice_text_Feedback.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    practice_text_Feedback.tStop = t  # not accounting for scr refresh
                    practice_text_Feedback.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'practice_text_Feedback.stopped')
                    # update status
                    practice_text_Feedback.status = FINISHED
                    practice_text_Feedback.setAutoDraw(False)
            
            # *practice_text_treatCounter* updates
            
            # if practice_text_treatCounter is starting this frame...
            if practice_text_treatCounter.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                practice_text_treatCounter.frameNStart = frameN  # exact frame index
                practice_text_treatCounter.tStart = t  # local t and not account for scr refresh
                practice_text_treatCounter.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(practice_text_treatCounter, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'practice_text_treatCounter.started')
                # update status
                practice_text_treatCounter.status = STARTED
                practice_text_treatCounter.setAutoDraw(True)
            
            # if practice_text_treatCounter is active this frame...
            if practice_text_treatCounter.status == STARTED:
                # update params
                pass
            
            # if practice_text_treatCounter is stopping this frame...
            if practice_text_treatCounter.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > practice_text_treatCounter.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    practice_text_treatCounter.tStop = t  # not accounting for scr refresh
                    practice_text_treatCounter.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'practice_text_treatCounter.stopped')
                    # update status
                    practice_text_treatCounter.status = FINISHED
                    practice_text_treatCounter.setAutoDraw(False)
            
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
            for thisComponent in practice_FeedbackCodeComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "practice_FeedbackCode" ---
        for thisComponent in practice_FeedbackCodeComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('practice_FeedbackCode.stopped', globalClock.getTime())
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        
        # --- Prepare to start Routine "practice_ITI500" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('practice_ITI500.started', globalClock.getTime())
        # Run 'Begin Routine' code from practice_ITI500_LSL_code
        #screen_outlet.push_sample([screen_markers[5]]) #pushes ITI marker
        
        # keep track of which components have finished
        practice_ITI500Components = [practice_trial_ITI]
        for thisComponent in practice_ITI500Components:
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
        
        # --- Run Routine "practice_ITI500" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.5:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *practice_trial_ITI* updates
            
            # if practice_trial_ITI is starting this frame...
            if practice_trial_ITI.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                practice_trial_ITI.frameNStart = frameN  # exact frame index
                practice_trial_ITI.tStart = t  # local t and not account for scr refresh
                practice_trial_ITI.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(practice_trial_ITI, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'practice_trial_ITI.started')
                # update status
                practice_trial_ITI.status = STARTED
                practice_trial_ITI.setAutoDraw(True)
            
            # if practice_trial_ITI is active this frame...
            if practice_trial_ITI.status == STARTED:
                # update params
                pass
            
            # if practice_trial_ITI is stopping this frame...
            if practice_trial_ITI.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > practice_trial_ITI.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    practice_trial_ITI.tStop = t  # not accounting for scr refresh
                    practice_trial_ITI.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'practice_trial_ITI.stopped')
                    # update status
                    practice_trial_ITI.status = FINISHED
                    practice_trial_ITI.setAutoDraw(False)
            
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
            for thisComponent in practice_ITI500Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "practice_ITI500" ---
        for thisComponent in practice_ITI500Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('practice_ITI500.stopped', globalClock.getTime())
        # Run 'End Routine' code from practice_trial_num_code
        # Trial_num is used in code_TargetPresTiming
        practice_trial_num = trial_num + 1;
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.500000)
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 6.0 repeats of 'PracticeLoop'
    
    
    # --- Prepare to start Routine "practice_EndScreen" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('practice_EndScreen.started', globalClock.getTime())
    # keep track of which components have finished
    practice_EndScreenComponents = [practice_EndScreenText]
    for thisComponent in practice_EndScreenComponents:
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
    
    # --- Run Routine "practice_EndScreen" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 3.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *practice_EndScreenText* updates
        
        # if practice_EndScreenText is starting this frame...
        if practice_EndScreenText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            practice_EndScreenText.frameNStart = frameN  # exact frame index
            practice_EndScreenText.tStart = t  # local t and not account for scr refresh
            practice_EndScreenText.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(practice_EndScreenText, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'practice_EndScreenText.started')
            # update status
            practice_EndScreenText.status = STARTED
            practice_EndScreenText.setAutoDraw(True)
        
        # if practice_EndScreenText is active this frame...
        if practice_EndScreenText.status == STARTED:
            # update params
            pass
        
        # if practice_EndScreenText is stopping this frame...
        if practice_EndScreenText.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > practice_EndScreenText.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                practice_EndScreenText.tStop = t  # not accounting for scr refresh
                practice_EndScreenText.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'practice_EndScreenText.stopped')
                # update status
                practice_EndScreenText.status = FINISHED
                practice_EndScreenText.setAutoDraw(False)
        
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
        for thisComponent in practice_EndScreenComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "practice_EndScreen" ---
    for thisComponent in practice_EndScreenComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('practice_EndScreen.stopped', globalClock.getTime())
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-3.000000)
    
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
    
    # --- Prepare to start Routine "first_ITI" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('first_ITI.started', globalClock.getTime())
    # keep track of which components have finished
    first_ITIComponents = [first_ITI_fig]
    for thisComponent in first_ITIComponents:
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
    
    # --- Run Routine "first_ITI" ---
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
        for thisComponent in first_ITIComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "first_ITI" ---
    for thisComponent in first_ITIComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('first_ITI.stopped', globalClock.getTime())
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-0.500000)
    
    # set up handler to look after randomisation of conditions etc
    MainLoop = data.TrialHandler(nReps=11.0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('mainloop_conditions.xlsx'),
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
        # Run 'Begin Routine' code from LSL_CuePresentation_code
        # Pushes screen marker
        if colour == 'blue': 
            screen_outlet.push_sample([screen_markers[0]]) #pushes cue_win marker
        elif colour == 'yellow':
            screen_outlet.push_sample([screen_markers[1]]) #pushes cue_loss marker
        
        #counter used to push one button marker per trial, regardless of number of button presses.
        CuePres_marker_count = 0 
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
            # Run 'Each Frame' code from LSL_CuePresentation_code
            # 'Early' button marker sent if they press
            if 'right' in EarlyPressCue.keys or 'left' in EarlyPressCue.keys:
                if CuePres_marker_count == 0:
                    behav_outlet.push_sample([behav_markers[0]])
                    CuePres_marker_count += 1
            
                   
                
            
            
            
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
            MainLoop.addData('EarlyPressCue.rt', EarlyPressCue.rt)
            MainLoop.addData('EarlyPressCue.duration', EarlyPressCue.duration)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.250000)
        
        # --- Prepare to start Routine "Fixation2" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('Fixation2.started', globalClock.getTime())
        EarlyPressFixation.keys = []
        EarlyPressFixation.rt = []
        _EarlyPressFixation_allKeys = []
        # Run 'Begin Routine' code from LSL_fixation_code
        screen_outlet.push_sample([screen_markers[2]]) #pushes screen fixation marker
        fixation_marker_count = 0 # Used for pushing button presses
        
        # keep track of which components have finished
        Fixation2Components = [FixationScreen, EarlyPressFixation]
        for thisComponent in Fixation2Components:
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
        
        # --- Run Routine "Fixation2" ---
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
            # Run 'Each Frame' code from LSL_fixation_code
            # pushes button marker if they press early
            if 'right' in EarlyPressFixation.keys or 'left' in EarlyPressFixation.keys:
                if fixation_marker_count == 0:
                    behav_outlet.push_sample([behav_markers[0]])
                    fixation_marker_count += 1 #this avoids multiple markers to be sent in same trial
            
                   
                
            
            
            
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
            for thisComponent in Fixation2Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Fixation2" ---
        for thisComponent in Fixation2Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('Fixation2.stopped', globalClock.getTime())
        # check responses
        if EarlyPressFixation.keys in ['', [], None]:  # No response was made
            EarlyPressFixation.keys = None
        MainLoop.addData('EarlyPressFixation.keys',EarlyPressFixation.keys)
        if EarlyPressFixation.keys != None:  # we had a response
            MainLoop.addData('EarlyPressFixation.rt', EarlyPressFixation.rt)
            MainLoop.addData('EarlyPressFixation.duration', EarlyPressFixation.duration)
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
        # Calculates target presentation time, based on performance of last 6 trials.
        # If accuracy on last 6 is greater than 66%, target is presented 20ms shorter.
        # If accuract on last 6 is less than 66%, target is presented 50ms longer.
        if trial_num <= 5:
            target_pres_time = target_pres_time;
            thisExp.addData('target_pres_time', target_pres_time);
        else:
            last_6_acc = calibration_accuracy[trial_num-5:trial_num+1];
            acc_ratio = sum(last_6_acc) / len(last_6_acc);
            if acc_ratio <= 0.66: #target presented for 50 ms longer
                target_pres_time = target_pres_time + 0.05
                thisExp.addData('target_pres_time', target_pres_time);
            else: #target presented for 20ms shorter
                target_pres_time = target_pres_time - 0.02
                thisExp.addData('target_pres_time', target_pres_time);
        ButtonPressTarget.keys = []
        ButtonPressTarget.rt = []
        _ButtonPressTarget_allKeys = []
        # Run 'Begin Routine' code from LSL_TargetPres_code
        screen_outlet.push_sample([screen_markers[3]]) #pushes screen targetpres marker
        TargetPres_marker_count = 0 # used for button press markers, avoids multiple markers to be sent in same trial
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
                if tThisFlipGlobal > TargetPresentationScreen.tStartRefresh + target_pres_time-frameTolerance:
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
                if tThisFlipGlobal > ButtonPressTarget.tStartRefresh + target_pres_time-frameTolerance:
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
            # Run 'Each Frame' code from LSL_TargetPres_code
            # Pushes correct or incorrect button marker
            if 'right' in ButtonPressTarget.keys or 'left' in ButtonPressTarget.keys:
                if TargetPres_marker_count == 0:
                    if corr_button == 'left' and ButtonPressTarget.keys == 'left': # correct answer
                        behav_outlet.push_sample([behav_markers[1]])
                        TargetPres_marker_count += 1 
                    elif corr_button == 'right' and ButtonPressTarget.keys == 'right': # correct answer
                        behav_outlet.push_sample([behav_markers[1]])
                        TargetPres_marker_count += 1
                    else:
                        behav_outlet.push_sample([behav_markers[2]]) # incorrect answer
                        TargetPres_marker_count += 1
             
            
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
        if ButtonPressTarget.keys != None:  # we had a response
            MainLoop.addData('ButtonPressTarget.rt', ButtonPressTarget.rt)
            MainLoop.addData('ButtonPressTarget.duration', ButtonPressTarget.duration)
        # Run 'End Routine' code from codeFeedbacksaving
        feedbackver = [];
        # Used to decide feedback participants get based on cue (win or loss) and button press
        if EarlyPressCue.keys != None or EarlyPressFixation.keys != None and colour == "blue":
            feedbackver = "3"; # Early press win cue -> did not win treat.
            thisExp.addData('outcome_label', "Early")
            thisExp.addData('outcome_val', -2)
            calibration_accuracy.append(0)
        elif EarlyPressCue.keys != None or EarlyPressFixation.keys != None and colour == "yellow":
            feedbackver = "4"; # Early press loss cue -> lost treat.
            thisExp.addData('outcome_label', "Early")
            thisExp.addData('outcome_val', -2)
            calibration_accuracy.append(0)
        elif ButtonPressTarget.corr == 1 and colour == "blue":
            feedbackver = "1"; # Correct press win cue -> won treat.
            thisExp.addData('outcome_label', "Correct")
            thisExp.addData('outcome_val', 1)
            calibration_accuracy.append(1)
        elif ButtonPressTarget.corr == 1 and colour == "yellow":
            feedbackver = "2"; # Correct press loss cue -> did not lose treat.
            thisExp.addData('outcome_label', "Correct")
            thisExp.addData('outcome_val', 1)
            calibration_accuracy.append(1)
        elif ButtonPressTarget.corr == 0 and colour == "blue":
            feedbackver = "3"; # Early press win cue -> did not win treat.
            thisExp.addData('outcome_label', "Incorrect")
            thisExp.addData('outcome_val', -1)
            calibration_accuracy.append(0)
        else:
            feedbackver = "4"; # Early press loss cue -> lost treat.
            thisExp.addData('outcome_label', "Incorrect")
            thisExp.addData('outcome_val', -1)
            calibration_accuracy.append(0)
            
        # the Routine "TargetPresentation" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "FeedbackCode" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('FeedbackCode.started', globalClock.getTime())
        # Run 'Begin Routine' code from feedbacktextcode
        if feedbackver == "1":
            text = "Du hast eine Süßigkeit gewonnen";
            textcolour = 'green';
            treat_counter += 1;
            
        elif feedbackver == "2":
            text = "Du hast keine Süßigkeit verloren";
            textcolour = 'green';
            
        elif feedbackver == "3":
            text = "Du hast die Süßigkeiten nicht gewonnen";
            textcolour = 'red';
        
        else:
            text = "Du hast eine Süßigkeit verloren";
            textcolour = 'red';
            treat_counter -= 1;
        
        
        text_Feedback.setColor(textcolour, colorSpace='rgb')
        text_Feedback.setText(text)
        text_treatCounter.setText(treat_counter)
        # Run 'Begin Routine' code from LSL_feedback_code
        screen_outlet.push_sample([screen_markers[4]]) #pushes screen fixation marker
        
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
        # Run 'Begin Routine' code from LSL_ITI500_code
        screen_outlet.push_sample([screen_markers[5]]) #pushes screen ITI marker
        
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
    # completed 11.0 repeats of 'MainLoop'
    
    
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
