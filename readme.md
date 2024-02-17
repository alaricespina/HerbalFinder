# HERBAL FINDER REPOSITORY

Repository containing the Source Code for the Herbal Finder Application

### Installation

#### Method 1
You can make use of the existing `.BAT` files in the project. These files are under the `./HerbalFinder` Folder and would setup the Node Modules needed for this project

The Order of execution is as follows:<br>
1. Open `./HerbalFinder/SETUP_NODE.bat` to setup the Node Modules
2. Open `./HerbalFinder/START_APP.bat` to start Expo Go and Launch the App

#### Method 2
You can also install the necessary modules for the application using the following commands

```
cd HerbalFinder
npm install
npx expo start
```

###### Note: If there are warnings regarding mismatched packages due to the different expected versions for each package for Expo and React Native you can run the following command:
```
npm install --fix
```

### Server Side Installation
The Server side is currently run by the repository owner and there is no need to install the necessary packages such as `Whisper`, `Torch` and the likes. Just Notify the repository owner to open the server

### <span style='color: red'>Warning </span><br>
Forcibly running the `REST_API_START_UP.bat` would not work due to the requirement of ngrok's Authentication Token which is only granted to the Repository Owner

###### Note: If you really want to run the server side, it is needed to modify the `REST_API_START_UP.bat` to your own domain and change the links in the `HerbalFinder/App.js`


