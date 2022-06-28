import React, {Component, useEffect} from 'react';

class ClassLogin extends Component{

    state = {
        credentials : {username : '', password : ''}
    }

    login = event => {
        console.log(this.state.credentials);
        fetch('http://127.0.0.1:8000/token',{
            method: 'POST',
            headers: {"content-type": "application/x-www-form-urlencoded"},
            body: JSON.stringify(`grant_type=&username=${this.state.credentials.username}&password=${this.state.credentials.password}&scope=&client_id=&client_secret=`),
            // body: JSON.stringify(this.state.credentials)
        })
        // .then(
        //     data => data.json()
        // )
        .then(
            data => {
                // data.JSON();
                console.log(data);
                (data.token ? window.location.href = './app' : window.location.href = './' )
            }
        )
        .catch(
            error => console.log(error)
        )
    }

    inputChange = event => {
        const cred = this.state.credentials;
        cred[event.target.name] = event.target.value;
        this.setState({credentials: cred});

    }


    
    render(){
        return(
            <div>
                <h1>Login component</h1>

                <label>
                    Username:
                    <input 
                    type="text" name="username"
                    value={this.state.credentials.username}
                    onChange={this.inputChange}
                    />
                </label>

                <br></br>

                <label>
                    Password:
                    <input 
                    type="password" name="password"
                    value={this.state.credentials.password}
                    onChange={this.inputChange}
                    />
                </label>
                <br/>
                <button onClick={this.login}>Login</button>
            </div>
        );
    }

}

export default ClassLogin;