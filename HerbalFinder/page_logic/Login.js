import { ngrok_link } from '../Consts'
import { Alert } from 'react-native';




const GetAccountAPI = async (username, password) => {
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

const MatchHandler = (response, navigation) => {
  if (response == undefined) {
    return
  }
  else {
    //console.log(response);
    if (response == true) {
      console.log("Successfully Logged In"); //Insert App Main Function Prompt
      ShowAlertMessage('Login', 'Successfully Logged In!');
      navigation.navigate("MainScreen")
    }
    else {
      console.log("Try Again"); //Insert Message in App to Try Again
      ShowAlertMessage('Login', 'Incorrect Username/Email or Password!\nPlease Try Again!');
    }
  }
}

const handleLogin = (username, password, navigation) => {
  console.log("Username: " + username + "\nPassword: " + password)

    // // Login Authentication Logic Using Local JSON
  GetAccountAPI(username, password).then((response) => {
    MatchHandler(response['match'], navigation);
  });
};

export default handleLogin;