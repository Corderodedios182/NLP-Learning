# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 17:24:08 2023

@author: corde
"""

#Los búferes de protocolo son el mecanismo extensible de Google,
#independiente del idioma y de la plataforma,
#para serializar datos estructurados;
#piense en XML, pero más pequeño, más rápido y más simple.

#Los protobufs son solo un reemplazo de la forma tradicional de transmitir mensajes usando XML o JSON.

#El siguiente código usa el mensaje del perfil como una clase y lo instancia.
#Luego asigna valores a sus atributos.
#Esto es similar a la forma en que instanciamos una clase en Python.

s = 'Johny'.SerializeToString()
# serialized output - b'\n\x05Johny\x10'

DESCRIPTOR = _descriptor.FileDescriptor(
  name='person.proto',
  package='',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x08person.proto\"#\n\x06Person\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x0b\n\x03\x61ge\x18\x02 \x01(\x03\x62\x06proto3'
)

