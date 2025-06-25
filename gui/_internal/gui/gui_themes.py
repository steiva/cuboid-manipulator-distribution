import dearpygui.dearpygui as dpg

def global_theme():
    with dpg.theme() as global_theme:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 6, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 6, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 6, category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive, (22,135,15), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_CheckMark, (22,135,15), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, (22,135,15), category=dpg.mvThemeCat_Core)


        with dpg.theme_component(dpg.mvHistogramSeries):
            dpg.add_theme_color(dpg.mvPlotCol_Fill, (0,255,0,50), category=dpg.mvThemeCat_Plots)

        with dpg.theme_component(dpg.mvButton, enabled_state=False):
            dpg.add_theme_color(dpg.mvThemeCol_Text, [85, 85, 85])
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, [51,51,55])
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, [51,51,55])
            # dpg.add_theme_color(dpg.mvPlotCol_Line, (245, 170, 66), category=dpg.mvThemeCat_Plots)
    dpg.bind_theme(global_theme)

def change_btn_clr(state):
    if state:
        with dpg.theme() as item_theme:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (25,170,40), category=dpg.mvThemeCat_Core)
        dpg.bind_item_theme('control_buttons', item_theme)
    else:
        with dpg.theme() as global_theme:
            dpg.bind_item_theme('control_buttons', global_theme)

def pickup_button_theme():
    with dpg.theme() as item_theme:
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, (255,0,0), category=dpg.mvThemeCat_Core)
    dpg.bind_item_theme('pos_pickup_button', item_theme)

NEW_ANCHOR_CREATION_TIPS = ('INSTALL CALIBRATION DISH.\nThis is a calibration setup function that is used to record the laser calibration points, or "anchors".'
                            ' The robot will go to these points when it begins the calibration process. You can use your keyboard to control the robot.'
                            ' Use the arrow keys to move the robot and press "s" to save a new anchor. In total, you are required to save four anchors,'
                            ' one for each black square on the calibration dish. It is recommended to position the laser dot in the center of the squares.'
                            ' Once you have saved all the four coordinates, press "esc" to exit this function and continue.')

ANCHORS_DETECTION_TIPS = ('INSTALL CALIBRATION DISH.\nThis is a calibration setup function that is used to record the black anchor'
                          ' regions on the special calibration culture dish. You should see a green'
                          ' outline around the inner side of the black squares. If there are four outlines'
                          ' seen and everything looks correct, you can safely exit this function and continue.')

SIZE_CONVERSION_TIPS = ('INSTALL CALIBRATION DISH.\nThis function allows to calculate the size conversion ratio needed to estimate' 
                        ' the actual physical size of the picked objects (effective diameter measured in microns).'
                        ' If you see a green outline around the black triangle in the center of the dish and everything looks correct'
                        ' you can safely exit this function and continue.')

CALIBRATION_TIPS = ('INSTALL CALIBRATION DISH.\nThe calibration function is used to bind the coordinate systems of the camera and the robotic arm.'
                    ' The function is largely automated: the robot will slide over the pre-defined positions (recorded in "Create new anchors" step)'
                    ' and record the laser pointer coordinates at each of the anchor regions (also previously defined in the "Anchor regions detection" step).'
                    ' If the program is succesful, it will exit and you can continue. If the laser is not detected at any of these steps, the function will'
                    ' exit with an error, saying "Laser not found". In this case, the user can try to redo previous steps or change the lighting conditions.')

AUTOMATIC_PROCEDURE_TIPS = ('INSTALL PETRI DISH WITH CUBOIDS.\nThis is the final step before a well plate can be filled with cuboids. The user needs to'
                            ' choose the type of well plate they want to fill, and can choose the particular wells to be filled by double-clicking on individual'
                            ' wells or by pressing down the LMB and draging across the well plate schematic to select multiple wells at a time.'
                            ' If the well-plate grid is unknown to the robot (in case when it has been physically moved and needs to be setup again),'
                            ' the well-plate calibration function can be used. Use your keyboard to approach the four corners of the well plate and align'
                            ' the pipette tip as close to the center of the well as possible. Press "s" to save the position. Once you are done recording'
                            ' all four positions, press "esc" to save the recorded coordinates. Your well plate is now calibrated. If you feel uncertain about the calibration'
                            ' select a couple wells on the schematic and press the "Test grid" button. This will trigger the robot to move over to the wells,'
                            ' allowing to check for any mistakes in the calibration. After this is finished, press the "Launch Automatic Procedure" button and'
                            ' proceed to the opened menu. The new menu presents the user with multiple picking options. Upon adjustment, press the red "Launch button".')