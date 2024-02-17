// Main
import {React, useState, useEffect, useRef} from 'react';

// Components
import { Switch, View, Image, ImageBackground, Text, TouchableOpacity, TextInput, Pressable, Alert, ActivityIndicator, ScrollView, FlatList, Touchable} from 'react-native';
import { StatusBar } from 'expo-status-bar';
import { LinearGradient } from 'expo-linear-gradient';
import MaterialCommunityIcons from 'react-native-vector-icons/MaterialCommunityIcons';
import MaterialIcons from 'react-native-vector-icons/MaterialIcons';
import Entypo from 'react-native-vector-icons/Entypo';
import MaskedView from '@react-native-masked-view/masked-view';
// import { Camera, CameraType, FlashMode } from 'expo-camera';
import * as MediaLibrary from 'expo-media-library'
import { Audio } from 'expo-av';
import { uploadAsync } from 'expo-file-system';
import { requestMediaLibraryPermissionsAsync, launchImageLibraryAsync } from 'expo-image-picker';
import { requestCameraPermissionsAsync, launchCameraAsync } from 'expo-image-picker';

// import { launchImageLibraryAsync, launchImageLibraryAsync, getCameraPermissionsAsync, getMediaLibraryPermissionsAsync } from 'expo-image-picker';




// ================================================================================================================
// Constants

const ngrok_link = "https://fit-krill-apparently.ngrok-free.app"

const defaultScreenStates = {
  Welcome : false,
  LoginSignUp : false,
  Home : false, 
  Search : false, 
  Scanner : false, 
  PostScan : false, 
  AccountBase : false,
  AccountProfile : false,
  AccountSettings : false, 
  AccountAbout : false,
  CameraConfirmation : false,
  DisplayPredictionResults : false
}

const defaultSettingStates = {
  notificationEnabled : false,
  emailEnabled : false,
  reminderEnabled : false, 
  locationEnabled : false
}



// ================================================================================================================
// Utiliy Functions

const MaterialCommunityGradientIcon = (iconName, gradientStart, gradientEnd, defaultColor, monitoringVariable) => {
  if (monitoringVariable == true) {
    return (

        <MaskedView className="flex-1 flex-row h-full w-full bg-transparent items-center justify-center"
        maskElement={
          <View className="w-full h-full items-center justify-center">
            <MaterialCommunityIcons name={iconName} size={24} color={"#0F0"}/>
          </View>            
        }
      >
        <LinearGradient 
        colors={[gradientStart, gradientEnd]} 
        className="w-full h-full items-center justify-center"
        start={{x:0.5, y:0}}
        end = {{x:0.5, y:0.9}}
        />
        </MaskedView>

      
    )
  } else {
    return (
      <View className="w-full h-full items-center justify-center">
        <MaterialCommunityIcons name={iconName} size={24} color={defaultColor}/>
      </View>
    )
  }
  
}

const MaterialGradientIcon = (iconName, gradientStart, gradientEnd, defaultColor, monitoringVariable) => {
  if (monitoringVariable == true) {
    return (
      <MaskedView className="flex-1 flex-row h-full w-full bg-transparent items-center justify-center"
        maskElement={
          <View className="w-full h-full items-center justify-center">
            <MaterialIcons name={iconName} size={24} color={"#0F0"}/>
          </View>            
        }
      >
        <LinearGradient 
        colors={[gradientStart, gradientEnd]} 
        className="w-full h-full items-center justify-center"
        start={{x:0.5, y:0}}
        end = {{x:0.5, y:0.9}}
        />
        </MaskedView>
    )
  } else {
    return (
      <View className="w-full h-full items-center justify-center">
        <MaterialIcons name={iconName} size={24} color={defaultColor}/>
      </View>
    )
  }
}

const ShowOKAlertMessage = (title, message) => {
  Alert.alert(title, message, [
    {text: 'OK', onPress: () => console.log('OK Pressed')},
  ]);
}

// ================================================================================================================
// Special Functions

// Take picture after camera button has been pressed




// Handle Camera Button pressed on Navigation Bar
const handleCameraPressed = (...args) => {
  console.log("Camera Pressed, Args: ")
  console.log(args)
  var [SetActiveScreen, DataObjects, SetDataObjects] = args[0]
  

  launchCameraAsync({
    allowsEditing : true,
    base64 : true
  }).then((resp) => {
    if (resp["assets"]) {
      var _x = {...DataObjects}
      _x.ImageURI = resp["assets"][0]["uri"]
      _x.ImageBase64 = resp["assets"][0]["base64"]
      SetDataObjects({..._x})
      console.log("Camera Result")
      console.log(resp["assets"][0]["uri"])
      console.log(_x.ImageBase64.length)
      AttemptPrediction([SetActiveScreen, DataObjects, SetDataObjects, _x.ImageBase64, _x.ImageURI])
    } else {
      console.log("Cancelled")
    }
  }).then(() => {
    
  })   
}

// Send the Login Request after login has been pressed
const sendLoginRequest = async(username, password) => {
  console.log("Logging in")
  console.log(ngrok_link.concat("/login"))
  const response = await fetch(ngrok_link.concat("/login"), {
    method: 'POST',
    headers: {
      Accept: 'application/json',
      'Content-Type': 'application/json',
      'ngrok-skip-browser-warning' : true
    },
    body: JSON.stringify({
      'username': username,
      'password': password
    })
  });
  return await response.json();
}

// Send the Sign Up Request after Signup has been pressed
const sendSignUpRequest = async(username, email, password) => {
  const response = await fetch(ngrok_link.concat('/signup'), {
    method: 'POST',
    headers: {
      Accept: 'application/json',
      'Content-Type': 'application/json',
      'ngrok-skip-browser-warning' : true
    },
    body: JSON.stringify({
      'username': username,
      'email': email,
      'password': password
    })
  });
  return await response.json();
}

// Synchronous handling of Login and Signup Request
const handleLoginSignUpAction = (mode, SetActiveScreen, DataObjects, SetDataObjects) => {
  var username = DataObjects.username 
  var email = DataObjects.email 
  var password = DataObjects.password 

  if (mode == "Login") {
    sendLoginRequest(username, password).then((response) => {
      console.log("Login Response: ")
      console.log(response)
      if (response["match"]) {
        var _x = {...DataObjects}
        _x.username = username 
        _x.password = password 
        _x.email = response["email"]
        SetDataObjects({..._x})
        handleMenuButtonsPressed([SetActiveScreen, "Home"])
      }
    })
  } else if (mode == "SignUp") {
    sendSignUpRequest(username, email, password).then((response) => {
      console.log("SignUp Response:")
      console.log(response)
      if (!(response["match"])) {
        sendLoginRequest(username, password).then((response) => {
          handleMenuButtonsPressed([SetActiveScreen, "Home"])
        })
      }
    })
  }
}

// Update Account Details (Unlock Fields) after button has been pressed
const updateAccountDetails = () => {

}

const ResetCapturedPhoto = (arg) => {
  var SetActiveScreen = arg
  handleMenuButtonsPressed([SetActiveScreen, "Scanner"])
}

const asyncSendImagePredictionRequest = async (Image) => {
  const response = await fetch(ngrok_link.concat('/predict_image'), {
    method: 'POST',
    headers: {
      Accept: 'application/json',
      'Content-Type': 'application/json',
      'ngrok-skip-browser-warning' : true
    },
    body: JSON.stringify({
      'input_image' : Image
    })
  })

  return await response.json()
}

const asyncSendNLPRequest = async (Text_Input) => {
  const response = await fetch(ngrok_link.concat('/predict_nlp'), {
    method: 'POST',
    headers: {
      Accept: 'application/json',
      'Content-Type': 'application/json',
      'ngrok-skip-browser-warning' : true
    },
    body: JSON.stringify({
      'input_text' : Text_Input
    })
  })
  
  return await response.json()
}

const asyncGetAudioTranscription = async (Audio_URI) => {
  const response = await uploadAsync(
    ngrok_link.concat('/convert_audio'),
    Audio_URI
    )
  return await JSON.parse(response.body)
}


const asyncGetPlantDetails = async (Plant_Name) => {
  const response = await fetch(ngrok_link.concat('/plant_data'), {
    method: 'POST',
    headers: {
      Accept: 'application/json',
      'Content-Type': 'application/json',
      'ngrok-skip-browser-warning' : true
    },
    body: JSON.stringify({
      'plant_name' : Plant_Name
    })
  })

  return await response.json()
}

const AttemptPrediction = (...args) => {
  var [SetActiveScreen, DataObjects, SetDataObjects, ib64, iURI] = args[0]

  var _x = {...DataObjects}
  _x.Predicting = true 
  _x.ImageURI = iURI
  console.log("Updating Image")
  console.log(DataObjects.ImageURI)
  SetDataObjects({..._x})

  console.log("Sending image")
  console.log(ib64.length)

  asyncSendImagePredictionRequest(ib64).then((response) => {
    console.log("Request has been processed, executing then block")
    console.log("Model Prediction:")
    m_predictions = response["predictions"]
    console.log(m_predictions[0])
  
    _x.Predicting = false
    _x.ImageBase64 = null 
    _x.Model_Predictions = m_predictions

    SetDataObjects({..._x})

    handleMenuButtonsPressed([SetActiveScreen, "DisplayPredictionResults"]) 
  })
}

// Recording functions

const playCreatedRecording = async (RecordedURI) => {
  
  const {sound} = await Audio.Sound.createAsync({
    uri : RecordedURI
  })
  console.log("Loaded Sound, Playing Sound")
  await sound.playAsync()
  console.log("Done Playing")
}

const asyncStartRecording = async (RecordingObject) => {
  if (RecordingObject) {
    console.log(RecordingObject)
    try {
      await RecordingObject.stopAndUnloadAsync()
    } catch (error) {
      console.log("Recording Object has already been unloaded")
    }
  }
  

  console.log("Starting Recording")
  var {recording} = await Audio.Recording.createAsync(Audio.RecordingOptionsPresets.HIGH_QUALITY)
  return recording
}

const asyncStopRecording = async (RecordingObject) => {
  console.log("Stopping Recording")
  await RecordingObject.stopAndUnloadAsync()
  return RecordingObject.getURI()
}

// ================================================================================================================
// Utility Functions

const handleMenuButtonsPressed = (...args) => {
  var [SetActiveScreen, targettedAttribute] = args[0]
  var dSS = {...defaultScreenStates}
  dSS[targettedAttribute] = true 
  SetActiveScreen({...dSS})
}

const displayObjectFull = (targetObject) => {
  for (var x in targetObject) {
    console.log(x + " : " + targetObject[x])  
  }
}

const handleScreenDisplay = (...args) => {

  var [ActiveScreen, SetActiveScreen, s_SwitchStates, set_s_SwitchStates, DataObjects, SetDataObjects] = args[0]

  if (ActiveScreen.Home) {
    return HomeScreen([SetActiveScreen, DataObjects, SetDataObjects])
  } else if (ActiveScreen.Search) {
    return SearchScreen([SetActiveScreen, DataObjects, SetDataObjects])
  } else if (ActiveScreen.Scanner) {
    return ScannerScreen([DataObjects, SetDataObjects])
  } else if (ActiveScreen.AccountBase) {
    return AccountBase([SetActiveScreen, DataObjects])
  } else if (ActiveScreen.AccountSettings) {
    return AccountSettings([SetActiveScreen, s_SwitchStates, set_s_SwitchStates])
  } else if (ActiveScreen.AccountAbout) {
    return AccountAboutUs([SetActiveScreen])
  } else if (ActiveScreen.AccountProfile) {
    return AccountProfile([DataObjects, SetDataObjects, SetActiveScreen])
  } else if (ActiveScreen.PostScan) {
    return PostScanScreen([SetActiveScreen, DataObjects, SetDataObjects])
  } else if (ActiveScreen.LoginSignUp) {
    return LoginAndSignUpScreen([SetActiveScreen, DataObjects, SetDataObjects])
  } else if (ActiveScreen.CameraConfirmation) {
    return CameraConfirmationScreen([SetActiveScreen, DataObjects, SetDataObjects])
  } else if (ActiveScreen.DisplayPredictionResults) {
    return DisplayPredictionResultsScreen([SetActiveScreen, DataObjects, SetDataObjects])
  
  } else {
    return (<></>)
  }

}

const HandleQuickSearch = (...args) => {
  var [SetActiveScreen, DataObjects, SetDataObjects, PlantTarget] = args[0]

  var _x = {...DataObjects}

  asyncGetPlantDetails(PlantTarget).then((response) => {
    response = response["text"]
    console.log("Get Plant Request Details:")
    console.log(response)
    _x.ImageURI = null 
    _x.Plant_Data = {
      Name : PlantTarget,
      Scientific_Name : response["Scientific Name"],
      Short_Info : response["Short Info"], 
      Usage : response["Usage"], 
      Cure : response["Cure"],
      Benefits : response["Benefits"],
      Sources : response["Sources"]
    }
    SetDataObjects({..._x})
    handleMenuButtonsPressed([SetActiveScreen, "PostScan"])
  })
}

const GetDefaultImageSource = (IN) => {
  var ImageName = IN

  switch (ImageName) {
    case "Ampalaya":
      return require("./pics/SampleLeafImages/Ampalaya/1.jpg")
      break

    case "Guava":
      return require("./pics/SampleLeafImages/Guava/1.jpg")
      break
    
    case "Jackfruit":
      return require("./pics/SampleLeafImages/Jackfruit/1.jpg")
      break

    case "Jasmine":
      return require("./pics/SampleLeafImages/Jasmine/1.jpg")
      break
    
    case "Lagundi":
      return require("./pics/SampleLeafImages/Lagundi/1.jpg")
      break

      
    case "Lemon":
      return require("./pics/SampleLeafImages/Lemon/1.jpg")
      break
    
    case "Malunggay":
      return require("./pics/SampleLeafImages/Malunggay/1.jpg")
      break

    case "Mango":
      return require("./pics/SampleLeafImages/Mango/1.jpg")
      break
    
    case "Mint":
      return require("./pics/SampleLeafImages/Mint/1.jpg")
      break

    case "Sambong":
      return require("./pics/SampleLeafImages/Sambong/1.jpg")
      break

    default:
      break
  }
}

const HandleSearchQuery = async (...args) => {
  var [DataObjects, SetDataObjects] = args[0]

  var text_input = DataObjects.SearchValue
  console.log("Text To be Searched:" + text_input)
  response = await asyncSendNLPRequest(text_input)
  console.log("NLP Result:")
  console.log(response)
  var _x = {...DataObjects}
  _x.Model_Predictions = response["predictions"]
  SetDataObjects({..._x})
  
}

const HandlePreSetSearchQuery = async (...args) => {
  var [DataObjects, SetDataObjects, text_input] = args[0]

  
  console.log("Text To be Searched:" + text_input)
  response = await asyncSendNLPRequest(text_input)
  console.log("NLP Result:")
  console.log(response)
  var _x = {...DataObjects}
  _x.Model_Predictions = response["predictions"]
  SetDataObjects({..._x})
  
}


// ================================================================================================================
// Navigation Bar Components

const MenuBarGalleryCircle = (...args) => {
  var [SetActiveScreen, DataObjects, SetDataObjects] = args[0]
  

  return (
    <>
      <TouchableOpacity className="z-10 absolute w-[calc(200/375*100%)] left-[calc(50%-200/375/2*100%)] aspect-square bg-[#1E1D1D] bottom-[-4%] rounded-full"
      onPress={() => {
        if (DataObjects.ImageURI || DataObjects.Model_Predictions){
          handleMenuButtonsPressed([SetActiveScreen, "PostScan"])
        }
      }
      }
      >
        <View className="bg-transparent w-full items-center h-full mt-3">
          <MaterialCommunityIcons name="image" size={25} color="#FFF"/>
        </View>        
      </TouchableOpacity>
    </>
  )
}

const NavBar = (...args) => {
  var [ActiveScreen, SetActiveScreen, DataObjects, SetDataObjects] = args[0]

  if (!(ActiveScreen.AccountSettings || 
    ActiveScreen.AccountAbout || 
    ActiveScreen.AccountProfile || 
    ActiveScreen.LoginSignUp ||
    ActiveScreen.CameraConfirmation)) {
    return (
      <>
        <View className="z-20 absolute w-[calc(90/375*100%)] aspect-square bg-[#090E05] left-[calc(50%-90/375/2*100%)] bottom-[2.75%] rounded-full">
        </View>
  
        <View className="z-30 absolute w-[calc(80/375*100%)] aspect-square left-[calc(50%-80/375/2*100%)] bottom-[3.5%] mt-0 rounded-full">
          <TouchableOpacity className="w-full h-full" onPress={() => {
            console.log("Camera Pressed")
            handleCameraPressed([SetActiveScreen, DataObjects, SetDataObjects])
            }}>
            <LinearGradient start={{x:0.25, y:0.25}} end = {{x:0.75, y:0.6}} colors={["#008000", "#2AAA8A"]} className="w-full h-full rounded-full items-center justify-center">
              <MaterialCommunityIcons name="camera" size={25} color="#000"/>
            </LinearGradient>        
          </TouchableOpacity>
        </View>
  
        <View className="z-10 absolute left-0 right-0 bg-[#2F2F2F] w-full h-[calc(75/812*100%)] bottom-0 rounded-full flex-row">
          <TouchableOpacity className="w-1/5 bg-transparent items-center justify-center" onPress={() => {
            console.log("Home Selected")
            handleMenuButtonsPressed([SetActiveScreen, "Home"])
            }}>
            <View className="w-full h-full">
              {MaterialCommunityGradientIcon("home-variant", "#75E00A", "#0AE0A0", "#FFF", ActiveScreen.Home)}
            </View>
          </TouchableOpacity>
          <TouchableOpacity className="w-1/5 bg-transparent items-center justify-center" onPress={() => {
            console.log("Search Selected")
            handleMenuButtonsPressed([SetActiveScreen, "Search"])
            }}>
            <View className="w-full h-full">
              {MaterialCommunityGradientIcon("magnify", "#75E00A", "#0AE0A0", "#FFF", ActiveScreen.Search)}
            </View>
          </TouchableOpacity>
          <View className="w-1/5 bg-transparent">
          </View>
          <TouchableOpacity className="w-1/5 bg-transparent items-center justify-center" onPress={() => {
            console.log("Scanner Selected")
            handleCameraPressed([SetActiveScreen, DataObjects, SetDataObjects])
          }}>
            <View className="w-full h-full">
              {MaterialCommunityGradientIcon("line-scan", "#75E00A", "#0AE0A0", "#FFF", ActiveScreen.Scanner)}
            </View>
          </TouchableOpacity>
          <TouchableOpacity className="w-1/5 bg-transparent items-center justify-center" onPress={() => {
            console.log("Account Selected")
            handleMenuButtonsPressed([SetActiveScreen, "AccountBase"])
            }}>
            <View className="w-full h-full">
              {MaterialGradientIcon("person", "#75E00A", "#0AE0A0", "#FFF", ActiveScreen.AccountBase)}
            </View>
          </TouchableOpacity>
        </View>
      </>
    )
  } else {
    <>
    </>
  }
  
}

// ================================================================================================================
// Screens
// Login And Signup Screen

// Mini Screen
const DisplayPredictionResultsScreen = (...args) => {
  var [SetActiveScreen, DataObjects, SetDataObjects] = args[0]

  return (
    <>
    <View className="absolute z-100 w-full h-full bg-[#090E05] items-center">
      <View className="absolute top-[5%]">
        <Text className="text-2xl text-white">Prediction Results</Text>
      </View>
      <View className="absolute top-[69%] h-[7.5%] w-5/6 items-center justify-center rounded-3xl bg-[#008000]">
        <TouchableOpacity onPress={() => {
          handleMenuButtonsPressed([SetActiveScreen, "Home"])
        }}>
          <Text className="text-white">Back to Home</Text>
        </TouchableOpacity>
      </View>
      

      <View className="absolute top-[13%] w-5/6 h-[55%] bg-transparent">
        <ScrollView className="h-full">
          {DataObjects.Model_Predictions.map((plant, index) => (
            <View key={index} className="w-full h-24 mt-[2.5%]">
              <TouchableOpacity onPress={() => {
                HandleQuickSearch([SetActiveScreen, DataObjects, SetDataObjects, plant])
              }} className="w-full h-full">
                <View className="w-full h-full items-center justify-center ">
                  <View className="w-full h-full absolute z-10">
                    <Image className="w-full h-full bg-transparent" source={GetDefaultImageSource(plant)}/>
                  </View>
                  <View className="w-full h-full bg-[#00000099] z-20">

                  </View>
                  <Text className="absolute z-30 text-white text-xl">{plant}</Text>
                </View>
              </TouchableOpacity>
            </View>
            
            ))
          }
        </ScrollView>
        
      </View>
      

      
    </View>
    </>
  )
}

const LoginAndSignUpScreen = (...args) => {
  var [SetActiveScreen, DataObjects, SetDataObjects] = args[0]
  console.log("Login And Signup")
  console.log("Data Objects")
  console.log(DataObjects)

  const LoginSet = () => {
    return (
      <>
        <View className="absolute w-full h-[10%] top-[20%] items-center">
          <View className="w-4/5 h-full">
            <TextInput 
            placeholder="Enter email or username" 
            placeholderTextColor="#766F6F" 
            className="w-full bg-transparent h-full text-white border-b-2"
            value = {DataObjects.username}
            onChangeText={(text) => {
              var curr = {...DataObjects}
              curr.username = text
              SetDataObjects(curr)
            }}
            />
          </View>
        </View>

        <View className="absolute w-full h-[10%] top-[40%] items-center">
          <View className="w-4/5 h-full">
            <TextInput 
            placeholder="Password" 
            placeholderTextColor="#766F6F" 
            secureTextEntry 
            className="w-full bg-transparent h-full text-white border-b-2"
            value = {DataObjects.password}
            onChangeText={(text) => {
              var curr = {...DataObjects}
              curr.password = text
              SetDataObjects(curr)
            }}
            />
          </View>
        </View>
        
        {/* <View className="absolute w-full h-[10%] top-[50%] items-center">
          <View className="w-4/5 h-full items-end">
            <TouchableOpacity onPress={() => console.log("Forgot your password? You done goofed boy")}>
              <Text className="text-[#766F6F]">Forgot Password?</Text>
            </TouchableOpacity>
          </View>
        </View> */}


        <View className="absolute w-full h-[10%] top-[70%] items-center">
          <TouchableOpacity 
            className="bg-green-950 h-10 w-2/3 items-center rounded-md mb-3 ps-4 justify-center rounded-full"
            onPress= {() => {
              handleLoginSignUpAction("Login", SetActiveScreen, DataObjects, SetDataObjects)
              console.log("Login Pressed")
            }}>
            <Text className="text-white text-base font-bold">Login</Text>
          </TouchableOpacity>
        </View>
      </>
      
    )
    
  }

  const SignUpSet = () => {
    return (
      <>
        <View className="absolute w-full h-[10%] top-[20%] items-center">
          <View className="w-4/5 h-full">
            <TextInput 
            placeholder="Enter Username" 
            placeholderTextColor="#766F6F" 
            className="w-full bg-transparent h-full text-white border-b-2"
            value = {DataObjects.username}
            onChangeText={(text) => {
              var curr = {...DataObjects}
              curr.username = text
              SetDataObjects(curr)
            }}
            />
          </View>
        </View>

        <View className="absolute w-full h-[10%] top-[40%] items-center">
          <View className="w-4/5 h-full">
            <TextInput 
            placeholder="Enter Email" 
            placeholderTextColor="#766F6F" 
            className="w-full bg-transparent h-full text-white border-b-2"
            value = {DataObjects.email}
            onChangeText={(text) => {
              var curr = {...DataObjects}
              curr.email = text
              SetDataObjects(curr)
            }}
            />
          </View>
        </View>

        <View className="absolute w-full h-[10%] top-[60%] items-center">
          <View className="w-4/5 h-full">
            <TextInput 
            placeholder="Enter Password" 
            placeholderTextColor="#766F6F" 
            secureTextEntry 
            className="w-full bg-transparent h-full text-white border-b-2"
            value = {DataObjects.password}
            onChangeText={(text) => {
              var curr = {...DataObjects}
              curr.password = text
              SetDataObjects(curr)
            }}
            />
          </View>
        </View>
        
        


        <View className="absolute w-full h-[10%] top-[80%] items-center">
          <TouchableOpacity onPress= {() => {
            handleLoginSignUpAction("SignUp", SetActiveScreen, DataObjects, SetDataObjects)
            console.log("Signup Pressed")
            }} className="bg-green-950 h-10 w-2/3 items-center rounded-md mb-3 ps-4 justify-center rounded-full">
            <Text className="text-white text-base font-bold">SignUp</Text>
          </TouchableOpacity>
        </View>
      </>
      
    )
  }

  return (
    <>
      <ImageBackground source= {require("./pics/PNG_LoginSignUp.png")} className="absolute z-0 w-full h-full object-cover">
      </ImageBackground> 


      <View className="absolute w-full h-full bg-[#00000090] items-center">
        <View className="absolute z-10 w-[calc(450/375*100%)] aspect-square top-[calc(-225/812*100%)] left-[calc(-75/375*100%)]">
        <LinearGradient className="w-full h-full rounded-full" colors={["#00800075", "#00800075"]}></LinearGradient>
      </View>
      <View className="absolute z-10 w-[calc(467/375*100%)] h-[calc(450/812*100%)] top-[calc(-286/812*100%)] left-[calc(-179/375*100%)]">
        <LinearGradient className="w-full h-full rounded-full" colors={["#00800075", "#2AAA8A75"]}></LinearGradient>
      </View>
      <View className="absolute z-10 w-[calc(467/375*100%)] aspect-square top-[calc(648/812*100%)] left-[calc(131/375*100%)]">
        <LinearGradient className="w-full h-full rounded-full" colors={["#00800075", "#2AAA8A75"]}></LinearGradient>
      </View>
      <View className="absolute z-30 w-[150%] aspect-square top-[calc(700/812*100%)] left-[-40%]">
        <LinearGradient className="w-full h-full rounded-full" colors={["#00800075", "#2AAA8A75"]}></LinearGradient>
      </View>

      
      <View className="absolute z-20 h-[calc(134/812*100%)] top-[7.5%] w-full items-center">
        <Image source={require('./pics/HBLogo.png')} className=" h-full aspect-square object-scale-down rounded-xl ml-5"/>
      </View>

      <View className="absolute z-30 w-[calc(316/375*100%)] bg-[#2F2F2F] h-[calc(434/812*100%)] top-[calc(210/812*100%)] rounded-xl border-2 border-[#766F6F]">
        <View className="absolute top-[5%] w-full h-[10%] items-center">
          <View className="w-4/5 h-full">
            <View className={`absolute z-20 w-[calc(54%)] h-full rounded-full border-2 border-black ${DataObjects.loginSignUpState ? 'bg-[#008000]' : 'bg-[#2AAA8A]'}`}>
              <Pressable 
              className="w-full h-full items-center justify-center"
              onPress={() => {
                console.log("Login Pressed")
                console.log(DataObjects.loginSignUpState)
                var x = {...DataObjects}
                x.loginSignUpState = true
                SetDataObjects(x)
              }}
              >
                <Text className="text-white" >Login</Text>
              </Pressable>
            </View>
            <View className={`absolute z-10 w-[calc(60%)] left-[40%] h-full rounded-full border-2 border-black ${DataObjects.loginSignUpState ? 'bg-[#2AAA8A]' : 'bg-[#008000]'}`}>
              <Pressable 
              className="w-full h-full items-center justify-center"
              onPress={() => {
                console.log("SignUp Pressed")
                console.log(DataObjects.loginSignUpState)
                var x = {...DataObjects}
                x.loginSignUpState = false
                SetDataObjects(x)
              }}
              >
                <Text className="text-white" >Sign Up</Text>
              </Pressable>
            </View>
          </View>
        </View>

        {DataObjects.loginSignUpState ? LoginSet() : SignUpSet()}
        
        </View>
                
      </View>
          

       
    </>

  );
};

const RecordingBoxComponent = (...args) => {
  var [setActiveScreen, DataObjects, SetDataObjects] = args[0]

  const CurrentlyRecording = DataObjects.currentlyRecordingState

  return (
    <View className="absolute z-10 w-full h-full bg-[#00000080] items-center justify-center">
      <View className="z-20 bg-[#2F2F2F] w-[90%] h-1/5 justify-center items-center rounded-lg">
        <View className="h-[45%] w-full items-center">

        
          <View className="h-full w-5/6 flex-row">
            <TouchableOpacity onPress={() => {
              console.log("Back to Non Recording")
              var _x = {...DataObjects}
              _x.showAudioRecordingState = false
              _x.RecordingObjectBase64 = null 
              _x.RecordingObjectURI = null 
              _x.TranscribedAudioData = null 

              if (_x.currentlyRecordingState) {
                asyncStopRecording(_x.RecordingObject).then((response) => {})
                _x.currentlyRecordingState = false 
              }

              SetDataObjects({..._x})
            }}>
              <View className="h-full aspect-square bg-green-800 items-center justify-center rounded-l-lg">
                <Entypo name="back" size={36} color="#FFF"/>
                <Text className="text-xs text-white -mt-1">Back</Text>
              </View>
            </TouchableOpacity>
            
            {
              CurrentlyRecording ? 
              <TouchableOpacity onPress={() => {
                var _x = {...DataObjects}
                _x.currentlyRecordingState = false
                asyncStopRecording(_x.RecordingObject).then((response) => {
                  console.log("Stop Recording has been pressed, Recording URI:")
                  console.log(response)
                  _x.RecordedURI = response
                  SetDataObjects({..._x})
                })
              }}>
                <View className="h-full aspect-square bg-green-800 items-center justify-center">
                  <Entypo name="controller-stop" size={36} color="#FFF"/>
                  <Text className="text-xs text-white -mt-1">Stop</Text>
                </View>
              </TouchableOpacity> :

              <TouchableOpacity onPress={() => {
                var _x = {...DataObjects}
                _x.currentlyRecordingState = true
                asyncStartRecording(_x.RecordingObject).then((result) => {
                  console.log("Start Recording has been pressed")
                  _x.RecordingObject = result
                  SetDataObjects({..._x})
                })
              }}>
              <View className="h-full aspect-square bg-green-800 items-center justify-center">
                <MaterialCommunityIcons name="circle" size={36} color="#FFF"/>
                <Text className="text-xs text-white -mt-1">Start</Text>
              </View>
              </TouchableOpacity>
            }
            <TouchableOpacity onPress={() =>  {
              console.log("Attempting to play recording from")
              console.log(DataObjects.RecordedURI)
              if (DataObjects.RecordedURI) {
                playCreatedRecording(DataObjects.RecordedURI).then(() => {})
                console.log("Done Playing Recording")
              }
            }}>
              <View className="h-full aspect-square bg-green-800 items-center justify-center">
                <Entypo name="controller-play" size={42} color="#FFF"/>
                <Text className="text-xs text-white -mt-2">Play</Text>
              </View>
            </TouchableOpacity>

            <TouchableOpacity onPress={() => {
              console.log("Sending to Server")
              if (DataObjects.RecordedURI) {
                console.log("Sending Recording Object from Recorded" + DataObjects.RecordedURI)
                asyncGetAudioTranscription(DataObjects.RecordedURI).then((result) => {
                  var _x = {...DataObjects}
                  _x.TranscribedAudioData = result["result"]
                  console.log("Transcription Results")
                  console.log(result)
                  _x.showAudioRecordingState = false 
                  _x.SearchValue = result["result"]
                  SetDataObjects({..._x})
                })
                

                
              } else {
                console.log("No Recorded Objects yet")
              }
            }}>
              <View className="h-full aspect-square bg-green-800 items-center justify-center rounded-r-lg">
                <Entypo name="upload-to-cloud" size={36} color="#FFF"/>
                <Text className="text-xs text-white -mt-1">Upload</Text>
              </View>
            </TouchableOpacity>

          </View>
        </View>
        
        
          
          
      </View>
    </View>
  )
}

// Home Screen
const HomeScreen = (...args) => {
  var [SetActiveScreen, DataObjects, SetDataObjects] = args[0]

  const ShowRecordingBox = DataObjects.showAudioRecordingState
  

  const _perm = Audio.requestPermissionsAsync()

  const AvailablePlants = ["Jackfruit", "Sambong", "Lemon", "Jasmine", "Mango", "Mint", "Ampalaya", "Malunggay", "Guava", "Lagundi"]
  return (
    <>
      {ShowRecordingBox ? 
      RecordingBoxComponent([SetActiveScreen, DataObjects, SetDataObjects])
      : <></>
      }
      <View className="absolute w-full h-2/5 rounded-3xl" >
        <View className="absolute w-full h-full z-20 bg-[#00000080] rounded-3xl"></View>
        <Image className="absolute w-full h-full z-10 rounded-3xl" source={require("./pics/PNG_AboutUs.png")}></Image>
      </View>

      <View className="absolute w-80 top-16 left-1/2 -ml-40 flex-row ">
        <View className="ml-1 mr-6 h-full items-center justify-center">
          <MaterialCommunityIcons name="menu" size={25} color="#FFF"/>
        </View>      
        <Text className="h-full text-white">Hi {DataObjects.username}!</Text>
        <View className="absolute h-full right-12 items-center justify-center">
          <TouchableOpacity onPress={() => handleCameraPressed([SetActiveScreen, DataObjects, SetDataObjects])}>
            <MaterialCommunityIcons name="line-scan" size={25} color="#FFF"/>
          </TouchableOpacity>
        </View>
        
      </View>

      <View className="absolute w-80 h-12 top-32 left-1/2 -ml-40"> 
        
        <View className="absolute z-20 h-full items-center justify-center left-[3.5%]">
          <TouchableOpacity onPress={() => {
            console.log("Search Pressed")
            HandleSearchQuery([DataObjects, SetDataObjects]).then(() => {
              handleMenuButtonsPressed([SetActiveScreen, "DisplayPredictionResults"])
            })
          }}>
            <MaterialCommunityIcons  name="magnify" size={30} color="#FFF"/>
          </TouchableOpacity>  
        </View>
        <TextInput
        placeholder='Search'
        placeholderTextColor="white"
        value={DataObjects.SearchValue}
        onChangeText={(text) => {
          var x = {...DataObjects}
          x.SearchValue = text 
          SetDataObjects({...x})
        }}
        onSubmitEditing={() => {
          console.log("Search Query: ")
          console.log(DataObjects.SearchValue)
          HandleSearchQuery([DataObjects, SetDataObjects]).then(() => {
            handleMenuButtonsPressed([SetActiveScreen, "DisplayPredictionResults"])
          })
          
        }}

        className = "absolute border-2 w-full h-full border-white rounded-lg pl-12 pr-12 text-white"
        />
        <View className="absolute h-full items-center justify-center right-[5%]">
          <TouchableOpacity onPress={() => {
            console.log("Microphone Touched")
            var _x = {...DataObjects}
            _x.showAudioRecordingState = true 
            SetDataObjects({..._x})
          }}>
            <MaterialCommunityIcons  name="microphone" size={30} color="#FFF"/>
          </TouchableOpacity>          
        </View>
      </View>

      <View className="absolute w-80 left-1/2 -ml-40 top-52">
        <Text className="text-white"> Know more about Herbal Plants </Text>
      </View>

      <View className="absolute w-80 left-1/2 -ml-40 top-64 h-24 flex-row">
        <View className="w-24 bg-emerald-500 h-full rounded-lg">
          <TouchableOpacity className="w-full h-full" onPress={() => {HandleQuickSearch([SetActiveScreen, DataObjects, SetDataObjects, "Ampalaya"])}}>
            <Image source={require("./pics/SampleLeafImages/Ampalaya/1.jpg")} className="w-full h-full bg-blue-100 rounded-lg"
          />
          </TouchableOpacity>
        </View>
        <View className="w-24 bg-emerald-600 h-full rounded-lg ml-4">
          <TouchableOpacity className="w-full h-full" onPress={() => {HandleQuickSearch([SetActiveScreen, DataObjects, SetDataObjects, "Sambong"])}}>
            <Image source={require("./pics/SampleLeafImages/Sambong/1.jpg")} className="w-full h-full bg-blue-100 rounded-lg"
            />
          </TouchableOpacity>
        </View>
        <View className="w-24 bg-emerald-700 h-full rounded-lg ml-4">
          <TouchableOpacity className="w-full h-full" onPress={() => {HandleQuickSearch([SetActiveScreen, DataObjects, SetDataObjects, "Jackfruit"])}}>
            <Image source={require("./pics/SampleLeafImages/Jackfruit/1.jpg")} className="w-full h-full bg-blue-100 rounded-lg"
          />
          </TouchableOpacity>
          
        </View>
      </View>

      <View className="absolute w-80 left-1/2 -ml-40 top-96">
        <Text className="text-white"> Popular Herbal Plants</Text>
      </View>

      

      <View className="absolute w-full h-32 left-1/2 -ml-44 bottom-48">
        <ScrollView horizontal={true} className="h-full" showsHorizontalScrollIndicator={false}>
          {
            AvailablePlants.map((plant, idx) => (
              <TouchableOpacity key={idx} onPress={() => {HandleQuickSearch([SetActiveScreen, DataObjects, SetDataObjects, plant])}}>
                <View className="h-full bg-[#2F2F2F] w-36 items-center rounded-lg ml-4">
                  <View className="h-3/4 w-full p-2">
                    <Image className="w-full h-full" source={GetDefaultImageSource(plant)}/>
                  </View>
                  <Text className="text-white">{plant}</Text>
                </View>
              </TouchableOpacity>
              

            ))
          }
        </ScrollView>
      </View>

      
      
      
    </>
  )
}

// Post Scan Screen
const PostScanScreen = (...args) => {
  var [SetActiveScreen, DataObjects, SetDataObjects] = args[0]



  return (
    <>

      <View className="bg-green-300 w-full h-[calc(236/812*100%)] items-center justify-center">
        {DataObjects.ImageURI ? 
        <Image className="w-full h-full" source={{uri:DataObjects.ImageURI}}/> : 
        <Image className="w-full h-full" source={GetDefaultImageSource(DataObjects.Plant_Data.Name)}/>
        }
      </View>


      <View className="absolute w-full top-[30%] items-center justify-center">
        <Text className="text-white">{DataObjects.Plant_Data.Name}</Text>
        <Text className="text-white">{DataObjects.Plant_Data.Scientific_Name}</Text>
      </View>

      <View className="absolute w-full top-[40%] h-[10%] items-center">
        <View className="w-4/5 flex-row h-full gap-x-[4%] -ml-[19%]">
          <TouchableOpacity className="h-full aspect-square" onPress={() => {
            var _x = {...DataObjects}
            _x.PostScanState = "usage"
            SetDataObjects({..._x})
          }}>
            <View className="bg-green-800 h-full aspect-square items-center justify-center">
              <Entypo name="hand" size={32} color="#FFF"/>
              <Text className="text-white text-xs">Usage</Text>
            </View>
          </TouchableOpacity>
          <TouchableOpacity className="h-full aspect-square" onPress={() => {
            var _x = {...DataObjects}
            _x.PostScanState = "cure"
            SetDataObjects({..._x})
          }}>
            <View className="bg-green-800 h-full aspect-square items-center justify-center">
              <Entypo name="drop" size={32} color="#FFF"/>
              <Text className="text-white text-xs">Cure</Text>
            </View>
          </TouchableOpacity>
          <TouchableOpacity className="h-full aspect-square" onPress={() => {
            var _x = {...DataObjects}
            _x.PostScanState = "benefits"
            SetDataObjects({..._x})
          }}>
            <View className="bg-green-800 h-full aspect-square items-center justify-center">
              <Entypo name="price-ribbon" size={32} color="#FFF"/>
              <Text className="text-white text-xs">Benefits</Text>
            </View>
          </TouchableOpacity>
          <TouchableOpacity className="h-full aspect-square" onPress={() => {
            var _x = {...DataObjects}
            _x.PostScanState = "sources"
            SetDataObjects({..._x})
          }}>
            <View className="bg-green-800 h-full aspect-square items-center justify-center">
              <Entypo name="archive" size={32} color="#FFF"/>
              <Text className="text-white text-xs">Sources</Text>
            </View>
          </TouchableOpacity>
          

        </View>
        
      </View>

      <View className="absolute w-full top-[52%] h-[34%]">
        <View className="w-[93%] ml-[4%]">
          <ScrollView className="">
            <Text className="text-white text-justify text-[8px]">{DataObjects.Plant_Data.Short_Info}</Text>
            {
              DataObjects.PostScanState == "usage" ? <>
                <Text className="text-white font-bold text-lg">Usage</Text>
                <Text className="text-white text-xs text-justify">{DataObjects.Plant_Data.Usage}</Text>
              </> :
              DataObjects.PostScanState == "cure" ? <>
              <Text className="text-white font-bold text-lg">Cure</Text>
              <Text className="text-white text-xs text-justify">{DataObjects.Plant_Data.Cure}</Text>
              </> :
              DataObjects.PostScanState == "benefits" ? <>
              <Text className="text-white font-bold text-lg">Benefits</Text>
              <Text className="text-white text-xs text-justify">{DataObjects.Plant_Data.Benefits}</Text>
              </> :
              DataObjects.PostScanState == "sources" ? <>
              <Text className="text-white font-bold text-lg">Sources</Text>

              {DataObjects.Plant_Data.Sources.map((source, idx) => (
                <Text className="text-white text-xs text-justify" key={idx}>{source}</Text>  
              ))}
                
              </> : <></>
            }
          </ScrollView>
          

        </View>
      </View>
      
    
    </>
    
  )
}

// Confirmation Screen After Scan
const CameraConfirmationScreen = (...args) => {
  var [SetActiveScreen, DataObjects, SetDataObjects] = args[0]

  console.log("Received Image URI at Camera Confirmation Screen")
  console.log(DataObjects.ImageURI)

  const Image_URI_Display = DataObjects.ImageURI
  const button_sizes = 48

  return (
    <>
    {DataObjects.Predicting ? 
    <>
      <View className="absolute z-10 w-full h-full bg-[#00000080] items-center justify-center">
        <View className="bg-[#090E05] w-4/5 h-1/6 flex-row justify-center items-center">
          <ActivityIndicator className="h-full" size={72} color="#00FF00"/>
          <Text className="ml-[10%] text-white">Waiting for Predictions</Text>
        </View>
      </View>
    </> : 
    <></>}
    <View className="w-full h-full bg-black items-center justify-center">
      <View className="h-3/5 w-3/5 items-center justify-center object-contain">
        <Image className="w-full h-full object-contain" source={{uri:DataObjects.ImageURI}}>
        </Image>
      </View>
      <View className="w-3/5 h-1/6 items-center justify-center flex-row">
        <TouchableOpacity className="w-1/2 h-full items-center justify-center" onPress={() => ResetCapturedPhoto(SetActiveScreen)}>
          <MaterialCommunityIcons name="delete-circle-outline" size={button_sizes} color="#FFF"/>
        </TouchableOpacity>
        <TouchableOpacity className="w-1/2 h-full items-center justify-center" onPress={() => AttemptPrediction([SetActiveScreen, DataObjects, SetDataObjects])}>
          <MaterialCommunityIcons name="check-circle-outline" size={button_sizes} color="#FFF"/>
        </TouchableOpacity>
      </View>
    </View>
    

    </>
  )
}

// Scanner Screen
const ScannerScreen = (...args) => {
  var [DataObjects, SetDataObjects] = args[0]
  // const permission = MediaLibrary.requestPermissionsAsync(true)
  // const camera_perm = Camera.requestCameraPermissionsAsync()

  const _perm = requestCameraPermissionsAsync()

  console.log("Scanner Screen Called")
  
  return (
    <>
      <View className="absolute z-10 h-full w-full items-center justify-center">
      
        <LinearGradient 
          className="absolute z-30 w-full h-full"
          colors={["transparent", "#000"]}
          start={{x:0.5, y:0.65}}
          end={{x:0.5, y:1.0}}
        />
        {/* <Camera 
          ref={DataObjects.Camera_Obj}
          type={CameraType.back}
          flashMode={FlashMode.auto} 
          className="absolute z-20 h-full aspect-[3/4] items-center justify-center">

          
        </Camera> */}
        <Image source={require("./scan-line.png")}/>
    
        <Text className="absolute z-30 text-white top-[70%]"> Scanner </Text>
        
      </View>
    </>
  )
}

// Search Screen
const SearchScreen = (...args) => {
  var [SetActiveScreen, DataObjects, SetDataObjects] = args[0]
  const ShowRecordingBox = DataObjects.showAudioRecordingState
  
  const AvailablePlants = ["Jackfruit", "Sambong", "Lemon", "Jasmine", "Mango", "Mint", "Ampalaya", "Malunggay", "Guava", "Lagundi"]

  return (
    <>
      {ShowRecordingBox ? 
        RecordingBoxComponent([SetActiveScreen, DataObjects, SetDataObjects])
        : <></>
      }
      <View className="absolute w-80 h-12 top-12 left-1/2 -ml-40">   
        <View className="absolute h-full items-center justify-center ml-2">
          <MaterialCommunityIcons  name="magnify" size={30} color="#FFF"/>
        </View>
        <TextInput
        placeholder='Search'
        placeholderTextColor="white"
        className = "absolute border-2 w-full h-full border-white rounded-lg pl-12 text-white"
        />
        <View className="absolute h-full items-center justify-center right-[5%]">
          <TouchableOpacity onPress={() => {
            console.log("Microphone Touched")
            var _x = {...DataObjects}
            _x.showAudioRecordingState = true 
            SetDataObjects({..._x})
          }}>
            <MaterialCommunityIcons  name="microphone" size={30} color="#FFF"/>
          </TouchableOpacity>          
        </View>
      </View>
      
      <View className="absolute w-full h-20 left-1/2 -ml-44 top-32">
        <ScrollView horizontal={true} className="h-full" showsHorizontalScrollIndicator={false}>
          {
            AvailablePlants.map((plant, idx) => (
              <TouchableOpacity key={idx} onPress={() => {HandleQuickSearch([SetActiveScreen, DataObjects, SetDataObjects, plant])}}>
                <View key={idx} className="h-full bg-[#2F2F2F] w-36 items-center justify-center rounded-lg ml-4">
                  <Text key={idx} className="text-white">{plant}</Text>
                </View>
              </TouchableOpacity>
              

            ))
          }
        </ScrollView>
        
        
      </View>

      <View className="absolute w-80 h-44 left-1/2 -ml-40 top-60 flex-row">
        
        <View className="h-full bg-[#2F2F2F] w-36 rounded-lg">
          <TouchableOpacity className="w-full h-full"
          onPress={() => {
            const SearchValue = "Cough"
            HandlePreSetSearchQuery([DataObjects, SetDataObjects, SearchValue]).then(() => {
              handleMenuButtonsPressed([SetActiveScreen, "DisplayPredictionResults"])
            })
          }}>
            <View className="h-2/3 w-full bg-green-300 rounded-t-lg">
              <Image source={require("./pics/SampleLeafImages/Lagundi/1.jpg")} className="w-full h-full bg-blue-100 rounded-lg"/>
            </View>
            <View className="h-1/3 w-full items-center justify-center">
              <Text className="text-white">Cough</Text>
            </View>
          </TouchableOpacity>
        </View>
        <View className="h-full bg-[#2F2F2F] w-36 items-center justify-center rounded-lg ml-8 mt-8">
          <TouchableOpacity className="w-full h-full"
          onPress={() => {
            const SearchValue = "Infections"
            HandlePreSetSearchQuery([DataObjects, SetDataObjects, SearchValue]).then(() => {
              handleMenuButtonsPressed([SetActiveScreen, "DisplayPredictionResults"])
            })
          }}>
            <View className="h-2/3 w-full bg-green-500 rounded-t-lg">
              <Image source={require("./pics/SampleLeafImages/Mango/1.jpg")} className="w-full h-full bg-blue-100 rounded-lg"/>
            </View>
            <View className="h-1/3 w-full items-center justify-center">
              <Text className="text-white">Infections</Text>
            </View>
          </TouchableOpacity>
        </View>
      </View>

      <View className="absolute w-80 h-44 left-1/2 -ml-40 bottom-36 flex-row">
        <View className="h-full bg-[#2F2F2F] w-36 rounded-lg">
          <TouchableOpacity className="w-full h-full"
          onPress={() => {
            const SearchValue = "Hypertension"
            HandlePreSetSearchQuery([DataObjects, SetDataObjects, SearchValue]).then(() => {
              handleMenuButtonsPressed([SetActiveScreen, "DisplayPredictionResults"])
            })
          }}
          >
            <View className="h-2/3 w-full bg-green-700 rounded-t-lg">
              <Image source={require("./pics/SampleLeafImages/Sambong/1.jpg")} className="w-full h-full bg-blue-100 rounded-lg"/>
            </View>
            <View className="h-1/3 w-full items-center justify-center">
              <Text className="text-white">Hypertension</Text>
            </View>
          </TouchableOpacity>
        </View>
        <View className="h-full bg-[#2F2F2F] w-36 items-center justify-center rounded-lg ml-8 mt-8">
          <TouchableOpacity className="w-full h-full"
          onPress={() => {
            const SearchValue = "Diabetes"
            HandlePreSetSearchQuery([DataObjects, SetDataObjects, SearchValue]).then(() => {
              handleMenuButtonsPressed([SetActiveScreen, "DisplayPredictionResults"])
            })
          }}>
            <View className="h-2/3 w-full bg-green-900 rounded-t-lg">
              <Image source={require("./pics/SampleLeafImages/Guava/1.jpg")} className="w-full h-full bg-blue-100 rounded-lg"/>
            </View>
            <View className="h-1/3 w-full items-center justify-center">
              <Text className="text-white">Diabetes</Text>
            </View>
          </TouchableOpacity>
        </View>
      </View>
    </>
    
  )
}

// About Us Screen
const AccountAboutUs = (...args) => {

  var [SetActiveScreen] = args[0]
  return (
    <ImageBackground
      source={require('./pics/PNG_AboutUs.png')}
      // style={{ flex: 1, resizeMode: 'cover', justifyContent: 'center', backgroundColor: 'rgba(0,0,0,1.00)' }}
      className ="w-full h-full"
    >
      <View className="items-center justify-center">
        {/* Herbal Finder Logo */}
        <Image 
        source={require('./pics/HBLogo.png')} 
        style={{ width: 150, height: 120, marginBottom: 20}} 
        />

        <Text className="text-white text-3xl font-bold mb-6">About Us</Text>
        <View className="w-5/6 bg-[#2F2F2F90] rounded-2xl p-[5%] mb-[20%]">
          <Text className="text-white text-xs mb-4 text-justify">
            Welcome to our Herbal Finder App! We are here to provide valuable information
            and promote the use of herbal plants.
          </Text>
          <Text className="text-white text-xs mb-4 text-justify">
            Our mission is to educate users about various herbal plants, their benefits,
            and how they can be used for different purposes, especially in natural remedies.
          </Text>
          <Text className="text-white text-xs mb-4 text-justify">
            Do you have any questions or concerns? Feel free to reach out to us at HerbalFinder@gmail.com.
          </Text>
        </View>
      </View>

      {/* "Back to Home" button with darker green color */}
      <TouchableOpacity
        className="z-40 absolute w-full h-16 bg-[#008000] bottom-0 items-center justify-center"
        onPress={() => {
          handleMenuButtonsPressed([SetActiveScreen, "AccountBase"])
        }}
      >
        <Text className="text-black text-xl font-bold">Back to Home</Text>
      </TouchableOpacity>
    </ImageBackground>
  );
};

// Acount Base Screen
const AccountBase = (...args) => {
  var [SetActiveScreen, DataObjects] = args[0]

  return (
    <>
      <View className="absolute top-0 h-[calc(410/812*100%)] w-full left-0 right-0 bg-green-500 rounded-2xl">
        <View className="items-center pt-14"> 
          <View className="flex-row pb-3">
            <MaterialIcons name="person" size={42} color="#FFF"/>
            <Text className="text-white text-xl font-bold"> My Account </Text>
          </View>
          <View className="rounded-full bg-green-600 flex items-center justify-center">
            <MaterialIcons name="person" size={180} color="#FFF"/>
          </View>
          <Text className="font-bold text-xl text-white">{DataObjects.username}</Text>
          <Text className="font-bold text-xs text-white">{DataObjects.email}</Text>
        </View>
      </View>

      <View className="absolute top-[calc(55%)] h-[calc(90/812*100%)]  w-full items-center">
        <View className="w-4/5 h-full">
          <TouchableOpacity className="absolute h-full left-0 aspect-square items-center justify-center bg-[#2F2F2F80] rounded-2xl" onPress={() => {
            console.log("Settings Pressed")
            // const _ = {...defaultScreenStates}
            // _.AccountSettings = true 
            // SetActiveScreen({..._})
            }}>
            <MaterialIcons name="settings" size={40} color="#FFF"/>
            <Text className="text-white text-xs">Settings</Text>
          </TouchableOpacity>
          <TouchableOpacity className="absolute h-full left-[35.5%]  aspect-square items-center justify-center bg-[#2F2F2F80] rounded-2xl" onPress={() => {
            console.log("Profile Pressed")
            const _ = {...defaultScreenStates}
            _.AccountProfile = true 
            SetActiveScreen({..._})
            }}>
            <MaterialIcons name="person" size={40} color="#FFF"/>
            <Text className="text-white text-xs">Profile</Text>
          </TouchableOpacity>
          <TouchableOpacity className="absolute h-full left-[71%] aspect-square items-center justify-center bg-[#2F2F2F80] rounded-2xl" onPress={() => {
            console.log("About Us Pressed")
            const _ = {...defaultScreenStates}
            _.AccountAbout = true 
            SetActiveScreen({..._})
            }}>
            <MaterialIcons name="help" size={40} color="#FFF"/>
            <Text className="text-white text-xs">About Us</Text>
          </TouchableOpacity>
        </View>
      </View>
      

      <View className="absolute top-[calc(68%)] items-center w-full h-[41%]">
        <View className="h-2/5 w-4/5 bg-[#2F2F2F80] rounded-2xl items-center">
          <Text className="text-white text-s font-medium">Your Favorite Plants</Text>
          {/* <View className="w-4/5 h-3/5 mt-2">
            <View className="absolute left-0 items-center w-1/3 h-full">
              <TouchableOpacity className="h-3/4 aspect-square bg-red-500 rounded-full"></TouchableOpacity>
              <Text className="text-white text-xs italic">Text 1</Text>
            </View>
            
            <View className="absolute left-[33.5%] items-center w-1/3 h-full">
              <TouchableOpacity className="h-3/4 aspect-square bg-blue-500 rounded-full"></TouchableOpacity>
              <Text className="text-white text-xs italic">Text 2</Text>
            </View>
            
            <View className="absolute left-[67%] items-center w-1/3 h-full">
              <TouchableOpacity className="h-3/4 aspect-square bg-green-500 rounded-full"></TouchableOpacity>
              <Text className="text-white text-xs italic">Text 3</Text>
            </View>
          </View> */}
        </View>
      </View>






      </>
  );
};

// Account Profile Screen
const AccountProfile = (...args) => {
  console.log("Profile Screen")
  var [DataObjects, SetDataObjects, SetActiveScreen] = args[0]


  return (
      <>
      
      <View className="absolute z-0 w-full h-1/2 top-0 bg-green-300">

      </View>
      <LinearGradient className="absolute z-10 w-full h-full"
      colors={["transparent", "#000"]}
      start={{x:0.5, y:0.0}}
      end={{x:0.5, y:0.3}}
      />
      <View className="absolute z-20 w-full h-full items-center">
          <View className="absolute z-20 w-4/5 h-full items-center">
              <View className="absolute w-full top-[7%]">
                  <Text className="text-white text-xl">My Profile</Text>
              </View>
              <View className="absolute  w-full top-[15%] h-[calc(78/812*100%)]">
                  <Text className="text-white">Name</Text>
                  <TextInput className="w-full border-[#DBFFB7] border-2 rounded-xl h-[70%] pl-4 text-white"
                    value={DataObjects.username}>

                  </TextInput>
              </View>
              <View className="absolute  w-full top-[30%] h-[calc(78/812*100%)]">
                  <Text className="text-white">Email</Text>
                  <TextInput className="w-full border-[#DBFFB7] border-2 rounded-xl h-[70%] pl-4 text-white"
                    value={DataObjects.email}>

                  </TextInput>
              </View>
              <View className="absolute  w-full top-[45%] h-[calc(78/812*100%)]">
                  <Text className="text-white">Password</Text>
                  <TextInput className="w-full border-[#DBFFB7] border-2 rounded-xl h-[70%] pl-4 text-white"
                    value={DataObjects.password}
                    secureTextEntry
                    >

                  </TextInput>
              </View>

              <View className="absolute w-full top-[85%] h-[calc(47/812*100%)] rounded-xl">
                  <TouchableOpacity className="w-full h-full" onPress={() => {
                      console.log("Back to Home Pressed")
                      handleMenuButtonsPressed([SetActiveScreen, "AccountBase"])
                      }}>
                      <LinearGradient
                      colors={["#008000", "#0AE0A0"]}
                      className="w-full h-full items-center justify-center"
                      >
                          <Text> Back To Home</Text>
                      </LinearGradient>
                  </TouchableOpacity>
              </View>
              
          </View>
      </View>
      
      </>
  )
}

// Account Settings Screen
const AccountSettings = (...args) => {
  var [SetActiveScreen, s_SwitchStates, set_s_SwitchStates] = args[0]

  console.log("Settings Screen Loaded")
  console.log("Initial Args:" + args)

  return (
    <View className="w-full h-full bg-black">
      
      <View className="w-full h-1/4 bg-green-600">
        <Text className="mt-12 ml-3 text-white font-bold text-3xl">Settings</Text>
      </View>
      <View className="w-full h-1/2 flex-column mt-[2%]">
        <View className="w-full flex-row ml-[16%]">
          <MaterialCommunityIcons name="bell" size={35} color="#FFF"/>
          <View className="mr-5"></View>
          <View className="flex-column w-[40%]">
            <Text className="text-lg font-bold text-white">Notification</Text>
            <Text className="text-xs text-white font-light">Caption...</Text>
          </View>
        <Switch trackColor={{false: '#767577', true: '#0ae0a0'}}
      thumbColor={s_SwitchStates.notificationEnabled ? '#008000' : '#f4f3f4'}
      ios_backgroundColor="#3e3e3e"
      onValueChange={() => {
        var _ = {...s_SwitchStates}
        _.notificationEnabled = !_.notificationEnabled
        set_s_SwitchStates({..._})
      }}
      value={s_SwitchStates.notificationEnabled}/>
        </View>

        <View className="my-1.5"></View>

        <View className="w-full flex-row ml-[16%]">
          <MaterialCommunityIcons name="email" size={35} color="#FFF"/>
          <View className="mr-5"></View>
          <View className="flex-column w-[40%]">
            <Text className="text-lg font-bold text-white">Email Updates</Text>
            <Text className="text-xs text-white font-light">Caption...</Text>
          </View>
          <Switch trackColor={{false: '#767577', true: '#0ae0a0'}}
        thumbColor={s_SwitchStates.emailEnabled ? '#008000' : '#f4f3f4'}
        ios_backgroundColor="#3e3e3e"
        onValueChange={() => {
          var _ = {...s_SwitchStates}
          _.emailEnabled = !_.emailEnabled
          set_s_SwitchStates({..._})
        }}
        value={s_SwitchStates.emailEnabled}/>
        </View>

        <View className="my-1.5"></View>

        <View className="w-full flex-row ml-[16%]">
          <MaterialCommunityIcons name="reminder" size={35} color="#FFF"/>
          <View className="mr-5"></View>
          <View className="flex-column w-[40%]">
            <Text className="text-lg font-bold text-white">Reminders</Text>
            <Text className="text-xs text-white font-light">Caption...</Text>
          </View>
          <Switch trackColor={{false: '#767577', true: '#0ae0a0'}}
        thumbColor={s_SwitchStates.reminderEnabled ? '#008000' : '#f4f3f4'}
        ios_backgroundColor="#3e3e3e"
        onValueChange={() => {
          var _ = {...s_SwitchStates}
          _.reminderEnabled = !_.reminderEnabled
          set_s_SwitchStates({..._})
        }}
        value={s_SwitchStates.reminderEnabled}/>
        </View>

        <View className="my-1.5"></View>

        <View className="w-full flex-row ml-[16%]">
          <MaterialCommunityIcons name="navigation" size={35} color="#FFF"/>
          <View className="mr-5"></View>
          <View className="flex-column w-[40%]">
            <Text className="text-lg font-bold text-white">Share Location</Text>
            <Text className="text-xs text-white font-light">Caption...</Text>
          </View>
          <Switch trackColor={{false: '#767577', true: '#0ae0a0'}}
        thumbColor={s_SwitchStates.locationEnabled ? '#008000' : '#f4f3f4'}
        ios_backgroundColor="#3e3e3e"
        onValueChange={() => {
          var _ = {...s_SwitchStates}
          _.locationEnabled = !_.locationEnabled
          set_s_SwitchStates({..._})
        }}
        value={s_SwitchStates.locationEnabled}/>
        </View>
      </View>

      <TouchableOpacity className="w-[90%] ml-[5%] mr-[5%] py-2 rounded-lg bg-[#008000] mt-[20%] items-center justify-center"
      onPress={() => {
        console.log("Back to Home is Pressed")
        handleMenuButtonsPressed([SetActiveScreen, "AccountBase"])
      }}>
        <Text className="text-black text-xl font-bold">Back to Home</Text>
      </TouchableOpacity>

    </View>
  )
}

// ================================================================================================================
// App Function

const App = () => {
  
  const defaultDataObjects = {
      Camera_Obj : useRef(null),
      username : null,
      email : null,
      password : null,
      loginSignUpState : true,
      showAudioRecordingState : false,
      currentlyRecordingState : false,
      RecordingObject : null,
      RecordingObjectURI : null,
      RecordingObjectBase64 : null,
      ImageURI : null,
      ImageBase64 : null,
      Predicting : false,
      Model_Predictions : null,
      PostScanState : "usage",
      Plant_Data : {
        Name : "null",
        Scientific_Name : "null",
        Short_Info : "null", 
        Usage : "null", 
        Cure : "null",
        Benefits : "null",
        Sources : "null"
      },
      TranscribedAudioData : null,
      SearchValue : null
  }

  const [permissionResponse, requestPermission] = Audio.usePermissions();


  console.log("\n" + Date() + " - Compiled");

  var [s_SwitchStates, set_s_SwitchStates] = useState({
    ...defaultSettingStates
  })

  // Initial State
  var dSS = {...defaultScreenStates} 
  var [ActiveScreen, SetActiveScreen] = useState({
    ...dSS
  })

  var [DataObjects, SetDataObjects] = useState({
    ...defaultDataObjects
  })

  var [startUp, setStartUp] = useState(1)
  if (startUp) {
    handleMenuButtonsPressed([SetActiveScreen, "LoginSignUp"])
    setStartUp(0)
  }

  console.log("***********Active Screen***********")
  displayObjectFull(ActiveScreen)
  console.log("***********************************")

  return (
    //LoginScreen()
    <View className="flex relative w-full h-full bg-[#090E05]">

    {handleScreenDisplay([ActiveScreen, SetActiveScreen, s_SwitchStates, set_s_SwitchStates, DataObjects, SetDataObjects])}

    {ActiveScreen.Home || ActiveScreen.DisplayPredictionResults ? MenuBarGalleryCircle([SetActiveScreen, DataObjects, SetDataObjects]) : <></>}

    {NavBar([ActiveScreen, SetActiveScreen, DataObjects, SetDataObjects])}


    <StatusBar style='auto'/>
    </View>

    
  )
};

export default App;