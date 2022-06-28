import React, { useState } from "react";
import Form from "react-bootstrap/Form";
import Button from "react-bootstrap/Button";
import "./Login.css";
import { useNavigate } from "react-router-dom";

function Register() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");

  let navigate = useNavigate();

  function validateForm() {
    return email.length > 0 && password.length > 0;
  }

  var reDrectPage = () => {
    navigate("/");
    // localStorage.setItem("token", "logedOut");
  };

  function handleSubmit(event) {
    event.preventDefault();

    if (password == confirmPassword) {
      var details = {
        username: email,
        password: password,
        // 'grant_type': 'password'
      };

      console.log(details);

      fetch("http://localhost:8000/user", {
        method: "POST",
        headers: {
          Accept: "application/json",
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ username: email, password: password }),
      })
        .then((res) => res.json())
        .then((data) => {
          console.log("Result", data);
        });
    }
  }

  return (
    <div className="Login">
      <Form onSubmit={handleSubmit}>
        <h3>User Register Page</h3>
        <Form.Group size="lg" controlId="email">
          <Form.Label>User Name</Form.Label>
          <Form.Control
            autoFocus
            // type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
          />
        </Form.Group>
        <br></br>
        <Form.Group size="lg" controlId="password">
          <Form.Label>Password</Form.Label>
          <Form.Control
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />
        </Form.Group>

        <br></br>
        <Form.Group size="lg" controlId="password">
          <Form.Label>Password</Form.Label>
          <Form.Control
            type="password"
            value={confirmPassword}
            onChange={(e) => setConfirmPassword(e.target.value)}
          />
        </Form.Group>
        <br></br>
        <Button block type="submit">
          Submit
        </Button>

        <Button
              className="float-end"
              variant="outline-primary"
              onClick={reDrectPage}
            >
              Login
            </Button>

      </Form>
    </div>
  );
}

export default Register;
